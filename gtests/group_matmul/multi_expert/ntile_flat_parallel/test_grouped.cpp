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

/// @file test_grouped.cpp
/// @brief N-tile flat parallel on grouped `group_matmul_direct` calls: each
/// expert type, activation, fused down projection and weighted reduce against
/// the AUTO path; the phases it serves; the calls it refuses; its packs.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common/float16.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/group_matmul/common/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel_config.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/nr32_pack.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "moe_test_utils.hpp"

void reset_grp_matmul_caches();

namespace {

namespace mm = zendnnl::lowoha::matmul;
using data_type_t = zendnnl::common::data_type_t;
using status_t = zendnnl::error_handling::status_t;
using act_t = mm::grp_matmul_gated_act_t;
using moe_test_utils::AlgoEnvGuard;
using moe_test_utils::AutoDecodeAlgoOverride;
using moe_test_utils::AutoPromptAlgoOverride;
using moe_test_utils::ExecutionPathCaptureGuard;
using zendnnl::common::float16_t;

uint16_t to_bf16(const float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

float from_bf16(const uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

bool fp16_host() {
    return zendnnl::common::zendnnl_platform_info().get_avx512_f16_status();
}

/// One grouped call: its experts' type and shapes and what it carries.
struct grouped_case {
    data_type_t wei = data_type_t::bf16;
    /// DA8W8 with the caller's s8 rows instead of bf16 rows.
    bool src_s8 = false;
    /// Quantized weights with bf16 rows: quantize them per token (else the
    /// call is weight-only).
    bool dynamic_quant = true;
    act_t act = act_t::silu_and_mul;
    bool trans = false;
    bool bias = false;
    std::vector<int> M {3, 0, 5, 1};
    int K = 192;
    int N = 256;
    /// Weights per scale group along K: s4 weights, and s8 weights with
    /// `s8_groups`.
    int group = 64;
    bool s8_groups = false;
    /// Source scales per (row, weight group) rather than per row.
    bool src_groups = false;
    bool fused = false;
    int N_down = 160;
    /// Op1 into the library's rows (fused calls only).
    bool op1_internal = false;
    /// Op2 into the caller's source rows (fused calls only).
    bool op2_internal = false;
    /// Tokens of a weighted reduce of the final rows, each routed to `topk`
    /// operations (which sets M); 0 for none.
    int tokens = 0;
    int topk = 2;
    /// Op1's output stride; 0 for N.
    int ldc = 0;
    std::vector<mm::matmul_post_op> post_ops;
    /// Every operation's `matmul_params::weight_cache_type`.
    int weight_cache = 1;
    /// Multiplies the source values and the quantized weights' scales.
    float magnitude = 1.0f;
    /// The call's team size; 0 for the default.
    int threads = 0;
};

bool gated(const act_t act) {
    return act != act_t::none;
}

/// The weights' scale group size, 0 for one scale per column.
int weight_group(const grouped_case &c) {
    return c.wei == data_type_t::s4 || c.s8_groups ? c.group : 0;
}

/// Source scales per row of an operation.
int source_groups(const grouped_case &c) {
    return c.src_groups ? c.K / c.group : 1;
}

data_type_t out_type(const grouped_case &c) {
    return c.wei == data_type_t::f16 ? data_type_t::f16 : data_type_t::bf16;
}

bool quantized(const grouped_case &c) {
    return c.wei == data_type_t::s8 || c.wei == data_type_t::s4;
}

/// An expert weight `[K, N]` (`[N, K]` when `trans`) with its scales.
struct weight_t {
    std::vector<uint8_t> data;
    std::vector<float> scale;
};

weight_t make_weight(std::mt19937 &rng, const data_type_t wei, const int K,
        const int N, const bool trans, const int group, const float magnitude) {
    weight_t w;
    const auto at = [&](const int k, const int n) {
        return trans ? static_cast<size_t>(n) * K + k
                     : static_cast<size_t>(k) * N + n;
    };
    std::uniform_real_distribution<float> real(-0.25f, 0.25f);
    if (wei == data_type_t::bf16 || wei == data_type_t::f16) {
        w.data.resize(static_cast<size_t>(K) * N * 2);
        auto *p = reinterpret_cast<uint16_t *>(w.data.data());
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                p[at(k, n)] = wei == data_type_t::bf16
                        ? to_bf16(real(rng))
                        : float16_t::f32_to_f16_val(real(rng));
        return w;
    }
    std::uniform_real_distribution<float> scale(
            0.004f * magnitude, 0.012f * magnitude);
    if (wei == data_type_t::s8) {
        std::uniform_int_distribution<int> q(-127, 127);
        w.data.resize(static_cast<size_t>(K) * N);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                w.data[at(k, n)] = static_cast<uint8_t>(q(rng));
        w.scale.resize(static_cast<size_t>(group > 0 ? K / group : 1) * N);
        for (float &s : w.scale)
            s = scale(rng);
        return w;
    }
    std::uniform_int_distribution<int> q(-8, 7);
    w.data.assign(static_cast<size_t>(K) * N / 2, 0);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            const size_t i = at(k, n);
            const uint8_t v = static_cast<uint8_t>(q(rng) & 0xF);
            w.data[i / 2] |= (i % 2 == 0) ? v : static_cast<uint8_t>(v << 4);
        }
    }
    w.scale.resize(static_cast<size_t>(K / group) * N);
    for (float &s : w.scale)
        s = 4.0f * scale(rng);
    return w;
}

/// A bias row of `n` values in the call's output type.
std::vector<uint8_t> make_bias(
        std::mt19937 &rng, const data_type_t dt, const int n) {
    std::uniform_real_distribution<float> real(-0.5f, 0.5f);
    std::vector<uint8_t> b(static_cast<size_t>(n) * 2);
    auto *p = reinterpret_cast<uint16_t *>(b.data());
    for (int i = 0; i < n; ++i)
        p[i] = dt == data_type_t::f16 ? float16_t::f32_to_f16_val(real(rng))
                                      : to_bf16(real(rng));
    return b;
}

/// A grouped call's inputs, one entry per expert.
struct grouped_inputs {
    std::vector<int> M;
    int lda = 0;
    std::vector<std::vector<uint8_t>> src;
    /// One scale per row: the caller's for s8 rows, else written by the call.
    std::vector<std::vector<float>> src_scale;
    std::vector<weight_t> w13, w2;
    std::vector<std::vector<uint8_t>> bias13, bias2;
    /// Per (token, slot): the expert and its row.
    std::vector<std::pair<int, int>> slots;
    std::vector<float> topk_weights;
};

grouped_inputs make_inputs(const grouped_case &c, const unsigned seed = 7) {
    std::mt19937 rng(seed);
    grouped_inputs in;
    in.M = c.M;
    const int n = static_cast<int>(c.M.size());
    if (c.tokens > 0) {
        std::fill(in.M.begin(), in.M.end(), 0);
        std::uniform_real_distribution<float> w(0.1f, 1.0f);
        for (int t = 0; t < c.tokens; ++t) {
            for (int k = 0; k < c.topk; ++k) {
                const int e = (t + 2 * k) % n;
                in.slots.push_back({e, in.M[e]++});
                in.topk_weights.push_back(w(rng));
            }
        }
    }
    in.lda = std::max(c.K, c.op2_internal ? c.N_down : 0) + 8;
    const data_type_t dt = out_type(c);
    std::uniform_real_distribution<float> real(-c.magnitude, c.magnitude);
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> row_scale(0.005f, 0.02f);
    const int K2 = gated(c.act) ? c.N / 2 : c.N;
    for (int e = 0; e < n; ++e) {
        const size_t elems = static_cast<size_t>(in.M[e]) * in.lda;
        in.src_scale.emplace_back(
                static_cast<size_t>(std::max(in.M[e], 1)) * source_groups(c));
        if (c.src_s8) {
            in.src.emplace_back(elems);
            for (uint8_t &v : in.src.back())
                v = static_cast<uint8_t>(q(rng));
            for (float &s : in.src_scale.back())
                s = row_scale(rng);
        } else {
            in.src.emplace_back(elems * 2);
            auto *p = reinterpret_cast<uint16_t *>(in.src.back().data());
            for (size_t i = 0; i < elems; ++i)
                p[i] = dt == data_type_t::f16
                        ? float16_t::f32_to_f16_val(real(rng))
                        : to_bf16(real(rng));
        }
        in.w13.push_back(make_weight(
                rng, c.wei, c.K, c.N, c.trans, weight_group(c), c.magnitude));
        in.bias13.push_back(make_bias(rng, dt, c.N));
        if (c.fused) {
            in.w2.push_back(make_weight(rng, c.wei, K2, c.N_down, c.trans,
                    weight_group(c), c.magnitude));
            in.bias2.push_back(make_bias(rng, dt, c.N_down));
        }
    }
    return in;
}

/// What a call wrote: each operation's final rows (the columns Op1's
/// activation produced, or Op2's), then the reduce's rows.
struct grouped_result {
    status_t status = status_t::failure;
    std::string path;
    std::vector<float> values;
    /// The source scales a call that quantizes its rows wrote.
    std::vector<float> src_scales;
    /// A fused call's Op1 rows when they go to the caller's `dst`.
    std::vector<float> op1;
};

/// A grouped call's arguments, for a test to edit before the call.
struct call_args {
    std::vector<char> &layout;
    std::vector<bool> &transA, &transB, &wconst;
    std::vector<int> &M, &N, &K, &lda, &ldb, &ldc;
    std::vector<float> &alpha, &beta;
    std::vector<const void *> &src, &weight, &bias;
    std::vector<void *> &dst;
    std::vector<mm::matmul_params> &params;
    mm::grp_matmul_fused_moe_params &fused;
    /// Call N-tile flat parallel's grouped executor itself rather than the
    /// public API.
    bool &direct;
};

/// Sets a test hook for a scope.
template <typename T>
struct hook_pin {
    std::atomic<T> &hook;
    const T prev;
    hook_pin(std::atomic<T> &h, const T value)
        : hook(h), prev(h.exchange(value)) {}
    ~hook_pin() { hook.store(prev); }
    hook_pin(const hook_pin &) = delete;
    hook_pin &operator=(const hook_pin &) = delete;
};

/// Runs `c` on `in` under the current ALGO settings.  Operation i is expert
/// `order[i]` (every expert in turn when empty); operations from `active` on
/// are the call's prepack extras.  With `bare_extras`, the extras come as
/// frameworks pass them: their weights alone, without `params`, input-side
/// arguments or down scales.
grouped_result run(const grouped_case &c, const grouped_inputs &in,
        std::vector<int> order = {}, int active = -1,
        const bool bare_extras = false,
        const std::function<void(call_args &)> &mutate = {}) {
    if (order.empty()) {
        order.resize(in.M.size());
        std::iota(order.begin(), order.end(), 0);
    }
    const int all = static_cast<int>(order.size());
    const int n = active < 0 ? all : active;
    const data_type_t dt = out_type(c);
    const int out13 = gated(c.act) ? c.N / 2 : c.N;
    const int ldc = c.ldc > 0 ? c.ldc : c.N;
    const int D = c.fused ? c.N_down : out13;
    const int K2 = out13;
    const bool dynamic = quantized(c) && !c.src_s8 && c.dynamic_quant;

    std::vector<std::vector<uint8_t>> src(n), dst(n), dst_down(n);
    std::vector<std::vector<float>> src_scale(n);
    std::vector<char> layout(all, 'r');
    std::vector<bool> transA(all, false), transB(all, c.trans),
            wconst(all, true);
    std::vector<int> Mv(all, 0), Nv(all, c.N), Kv(all, c.K), lda(all, in.lda),
            ldb(all, c.trans ? c.K : c.N), ldcv(all, ldc);
    std::vector<float> alpha(all, 1.0f), beta(all, 0.0f);
    std::vector<const void *> srcv(all, nullptr), wei(all, nullptr),
            bias(all, nullptr);
    std::vector<void *> dstv;
    std::vector<mm::matmul_params> params(all);
    for (int i = 0; i < all; ++i) {
        const int e = order[i];
        if (i < n) {
            Mv[i] = in.M[e];
            src[i] = in.src[e];
            src_scale[i] = in.src_scale[e];
            srcv[i] = src[i].data();
        }
        wei[i] = in.w13[e].data.data();
        if (c.bias && i < n) bias[i] = in.bias13[e].data();
        mm::matmul_params &p = params[i];
        p.dtypes.wei = c.wei;
        p.dtypes.dst = dt;
        p.dtypes.src = c.src_s8 ? data_type_t::s8
                : quantized(c)  ? data_type_t::bf16
                                : c.wei;
        p.dtypes.bias = c.bias ? dt : data_type_t::none;
        p.weight_cache_type = c.weight_cache;
        p.num_threads = c.threads;
        p.postop_ = c.post_ops;
        if (quantized(c)) {
            p.dtypes.compute
                    = dynamic || c.src_s8 ? data_type_t::s8 : data_type_t::bf16;
            p.dynamic_quant = dynamic;
            p.quant_params.wei_scale.buff = in.w13[e].scale.data();
            p.quant_params.wei_scale.dt = data_type_t::f32;
            p.quant_params.wei_scale.dims = weight_group(c) > 0
                    ? std::vector<int64_t> {c.K / c.group, c.N}
                    : std::vector<int64_t> {1, c.N};
            if (i < n && (dynamic || c.src_s8)) {
                p.quant_params.src_scale.buff = src_scale[i].data();
                p.quant_params.src_scale.dt = data_type_t::f32;
                p.quant_params.src_scale.dims
                        = {std::max(in.M[e], 1), source_groups(c)};
            }
        }
    }
    if (n < all) {
        params[0].active_matmul = static_cast<uint32_t>(n);
        params[0].total_matmul = static_cast<uint32_t>(all);
    }
    if (c.op1_internal) {
        ldcv.clear();
    } else {
        // A row even without rows: callers pass every operation a buffer.
        for (int i = 0; i < n; ++i) {
            dst[i].assign(
                    static_cast<size_t>(std::max(Mv[i], 1)) * ldc * 2, 0xFF);
            dstv.push_back(dst[i].data());
        }
    }

    mm::grp_matmul_fused_moe_params fused;
    if (c.fused) {
        const std::vector<int64_t> down_dims = weight_group(c) > 0
                ? std::vector<int64_t> {K2 / c.group, c.N_down}
                : std::vector<int64_t> {1, c.N_down};
        for (int i = 0; i < all; ++i) {
            const int e = order[i];
            fused.down_weight.push_back(in.w2[e].data.data());
            fused.N_down.push_back(c.N_down);
            fused.ldb_down.push_back(c.trans ? K2 : c.N_down);
            fused.bias_down.push_back(
                    c.bias && i < n ? in.bias2[e].data() : nullptr);
            if (quantized(c)) {
                mm::grp_matmul_fused_moe_params::down_weight_quant_t s;
                s.buff = in.w2[e].scale.data();
                s.dt = data_type_t::f32;
                s.dims = down_dims;
                fused.down_scale.push_back(s);
            }
        }
        fused.bias_dt_down = c.bias ? dt : data_type_t::none;
        if (!c.op2_internal) {
            for (int i = 0; i < n; ++i) {
                dst_down[i].assign(
                        static_cast<size_t>(std::max(Mv[i], 1)) * c.N_down * 2,
                        0xFF);
                fused.dst_down.push_back(dst_down[i].data());
                fused.ldc_down.push_back(c.N_down);
            }
        }
    }

    // Row j of operation i's final output.
    const auto final_row = [&](const int i, const int j) -> uint16_t * {
        if (!c.fused) {
            return reinterpret_cast<uint16_t *>(dst[i].data())
                    + static_cast<size_t>(j) * ldc;
        }
        if (c.op2_internal) {
            return reinterpret_cast<uint16_t *>(src[i].data())
                    + static_cast<size_t>(j) * in.lda;
        }
        return reinterpret_cast<uint16_t *>(dst_down[i].data())
                + static_cast<size_t>(j) * c.N_down;
    };
    mm::group_matmul_moe_postop_params reduce;
    std::vector<const void *> row_ptrs;
    std::vector<uint8_t> reduced;
    if (c.tokens > 0) {
        for (const auto &slot : in.slots)
            row_ptrs.push_back(final_row(slot.first, slot.second));
        reduced.assign(static_cast<size_t>(c.tokens) * D * 2, 0xFF);
        reduce.num_tokens = c.tokens;
        reduce.topk = c.topk;
        reduce.output = reduced.data();
        reduce.ldc_output = D;
        reduce.topk_weights = in.topk_weights.data();
        reduce.row_ptrs = row_ptrs.data();
    }
    mm::grp_matmul_gated_act_params act;
    act.act = c.act;

    bool direct = false;
    if (mutate) {
        call_args args {layout, transA, transB, wconst, Mv, Nv, Kv, lda, ldb,
                ldcv, alpha, beta, srcv, wei, bias, dstv, params, fused,
                direct};
        mutate(args);
    }
    if (bare_extras) {
        layout.resize(n);
        transA.resize(n);
        Mv.resize(n);
        alpha.resize(n);
        beta.resize(n);
        srcv.resize(n);
        lda.resize(n);
        bias.resize(n);
        if (!ldcv.empty()) ldcv.resize(n);
        params.resize(n);
        if (c.fused && quantized(c)) fused.down_scale.resize(n);
    }
    grouped_result r;
    {
        ExecutionPathCaptureGuard capture;
        r.status = direct
                ? mm::ntile_flat_parallel::execute_grouped(Mv.size(), layout,
                          transA, transB, Mv, Nv, Kv, alpha, srcv, lda, wei,
                          ldb, bias, beta, dstv, ldcv, wconst, params, act.act,
                          c.fused ? &fused : nullptr,
                          c.tokens > 0 ? &reduce : nullptr, 1,
                          mm::ntile_flat_parallel::flush_count())
                : mm::group_matmul_direct(layout, transA, transB, Mv, Nv, Kv,
                          alpha, srcv, lda, wei, ldb, bias, beta, dstv, ldcv,
                          wconst, params, c.tokens > 0 ? &reduce : nullptr,
                          &act, c.fused ? &fused : nullptr);
        const char *path
                = mm::test_api::s_last_group_matmul_direct_execution_path.load(
                        std::memory_order_relaxed);
        r.path = path != nullptr ? path : "";
    }
    if (dynamic) {
        for (int i = 0; i < n; ++i) {
            r.src_scales.insert(r.src_scales.end(), src_scale[i].begin(),
                    src_scale[i].begin()
                            + static_cast<size_t>(Mv[i]) * source_groups(c));
        }
    }
    const auto value = [&](const uint16_t v) {
        return dt == data_type_t::f16 ? float16_t::f16_to_f32_val(v)
                                      : from_bf16(v);
    };
    for (int i = 0; c.fused && !c.op1_internal && i < n; ++i) {
        for (int j = 0; j < Mv[i]; ++j) {
            const auto *row = reinterpret_cast<const uint16_t *>(dst[i].data())
                    + static_cast<size_t>(j) * ldc;
            for (int d = 0; d < out13; ++d)
                r.op1.push_back(value(row[d]));
        }
    }
    for (int i = 0; i < n; ++i) {
        if (c.op1_internal && !c.fused) break;
        for (int j = 0; j < Mv[i]; ++j) {
            const uint16_t *row = final_row(i, j);
            for (int d = 0; d < D; ++d)
                r.values.push_back(value(row[d]));
        }
    }
    const auto *out = reinterpret_cast<const uint16_t *>(reduced.data());
    for (size_t i = 0; i < reduced.size() / 2; ++i)
        r.values.push_back(value(out[i]));
    return r;
}

/// `got` is `want` within `tol` of `want`'s largest magnitude.
void expect_close(const grouped_result &got, const grouped_result &want,
        const double tol) {
    ASSERT_FALSE(want.values.empty());
    ASSERT_EQ(got.values.size(), want.values.size());
    double peak = 0.0, worst = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < want.values.size(); ++i) {
        peak = std::max(peak, std::fabs(static_cast<double>(want.values[i])));
        const double err = std::fabs(
                static_cast<double>(got.values[i]) - want.values[i]);
        if (!(err <= worst)) {
            worst = std::isnan(err) ? INFINITY : err;
            at = i;
        }
    }
    EXPECT_LE(worst, tol * std::max(peak, 1e-3))
            << "value " << at << ": " << got.values[at] << " vs "
            << want.values[at] << " (peak " << peak << ")";
}

std::string activation_label(const act_t act) {
    switch (act) {
        case act_t::silu_and_mul: return "silu";
        case act_t::gelu_and_mul: return "gelu";
        case act_t::swiglu_oai_mul: return "swiglu_oai";
        default: return "none";
    }
}

/// Runs `c` under N-tile flat parallel and under AUTO and compares them.  FP16
/// needs AVX-512-FP16, without which the case passes untested.
void check_against_auto(const grouped_case &c) {
    if (c.wei == data_type_t::f16 && !fp16_host()) return;
    const grouped_inputs in = make_inputs(c);
    reset_grp_matmul_caches();
    grouped_result requested;
    {
        AlgoEnvGuard pin(4);
        requested = run(c, in);
    }
    ASSERT_EQ(requested.status, status_t::success);
    EXPECT_EQ(requested.path, "ntile_flat_parallel");
    reset_grp_matmul_caches();
    grouped_result automatic;
    {
        AlgoEnvGuard pin(0);
        automatic = run(c, in);
    }
    ASSERT_EQ(automatic.status, status_t::success);
    EXPECT_NE(automatic.path, "ntile_flat_parallel");
    // AUTO applies the activation to rows already rounded to bf16, so a
    // fused quantized call quantizes a slightly different intermediate.
    const bool requantized = c.fused && quantized(c);
    const double tol = c.wei == data_type_t::f16 || requantized ? 0.03 : 0.02;
    expect_close(requested, automatic, tol);
    if (!automatic.op1.empty()) {
        // Op1's rows in the caller's `dst` are AUTO's too.
        grouped_result got, want;
        got.values = requested.op1;
        want.values = automatic.op1;
        expect_close(got, want, 0.02);
    }
    // Rows the call quantizes get the scales AUTO gives them, per row or per
    // (row, group).
    ASSERT_EQ(requested.src_scales.size(), automatic.src_scales.size());
    for (size_t i = 0; i < requested.src_scales.size(); ++i) {
        EXPECT_NEAR(requested.src_scales[i], automatic.src_scales[i],
                1e-6f * std::fabs(automatic.src_scales[i]))
                << "source scale " << i;
    }
    reset_grp_matmul_caches();
}

struct type_case {
    data_type_t wei;
    bool src_s8;
    const char *name;
};

const type_case kTypes[] = {
        {data_type_t::bf16, false, "bf16"},
        {data_type_t::f16, false, "f16"},
        {data_type_t::s8, false, "da8w8"},
        {data_type_t::s8, true, "da8w8_s8_rows"},
        {data_type_t::s4, false, "w4a8"},
};

grouped_case case_of(const type_case &t) {
    grouped_case c;
    c.wei = t.wei;
    c.src_s8 = t.src_s8;
    return c;
}

/// Pins what these tests assume of the process, whatever its environment
/// sets: weight cache mode 1 and no phase setting.
class NTileFlatParallelGrouped : public ::testing::Test {
protected:
    zendnnl::common::matmul_config_t &config_
            = zendnnl::common::matmul_config_t::instance();
    const int32_t mode_ = config_.get_weight_cache();
    AutoDecodeAlgoOverride decode_ {7};
    AutoPromptAlgoOverride prompt_ {7};
    NTileFlatParallelGrouped() { config_.set_weight_cache(1); }
    ~NTileFlatParallelGrouped() override { config_.set_weight_cache(mode_); }
};

TEST_F(NTileFlatParallelGrouped, MatchesAutoForEachTypeAndActivation) {
    for (const type_case &t : kTypes) {
        for (const act_t act : {act_t::none, act_t::silu_and_mul,
                     act_t::gelu_and_mul, act_t::swiglu_oai_mul}) {
            for (const bool trans : {false, true}) {
                if (t.wei == data_type_t::s4 && trans) continue;
                SCOPED_TRACE(std::string(t.name) + " " + activation_label(act)
                        + (trans ? " trans" : ""));
                grouped_case c = case_of(t);
                c.act = act;
                c.trans = trans;
                ASSERT_NO_FATAL_FAILURE(check_against_auto(c));
            }
        }
    }
}

TEST_F(NTileFlatParallelGrouped, MatchesAutoWithBiasAndPromptRows) {
    for (const type_case &t : kTypes) {
        SCOPED_TRACE(t.name);
        grouped_case c = case_of(t);
        c.bias = true;
        c.M = {70, 3, 0, 129};
        ASSERT_NO_FATAL_FAILURE(check_against_auto(c));
    }
}

TEST_F(NTileFlatParallelGrouped, FusedMatchesAutoInEveryOutputMode) {
    for (const type_case &t : kTypes) {
        // 0: the caller's rows for both outputs; 1: the library's Op1 rows;
        // 2: the library's Op1 rows and Op2 into the source rows.
        for (const int mode : {0, 1, 2}) {
            for (const act_t act : {act_t::none, act_t::silu_and_mul,
                         act_t::gelu_and_mul, act_t::swiglu_oai_mul}) {
                if (mode == 2 && t.src_s8) continue;
                SCOPED_TRACE(std::string(t.name) + " mode "
                        + std::to_string(mode) + " " + activation_label(act));
                grouped_case c = case_of(t);
                c.fused = true;
                c.bias = true;
                c.act = act;
                c.op1_internal = mode >= 1;
                c.op2_internal = mode == 2;
                ASSERT_NO_FATAL_FAILURE(check_against_auto(c));
                // Prompt-shaped rows: an operation's rows past its first.
                c.M = {70, 3, 0, 129};
                ASSERT_NO_FATAL_FAILURE(check_against_auto(c));
            }
        }
    }
}

TEST_F(NTileFlatParallelGrouped, WeightedReduceMatchesAuto) {
    for (const type_case &t : kTypes) {
        for (const bool fused : {false, true}) {
            SCOPED_TRACE(std::string(t.name) + (fused ? " fused" : ""));
            grouped_case c = case_of(t);
            c.fused = fused;
            c.M.assign(6, 0);
            c.tokens = 9;
            c.topk = 3;
            ASSERT_NO_FATAL_FAILURE(check_against_auto(c));
        }
    }
}

// Per-group weight scales on s8 and s4, against one source scale per row and
// one per (row, weight group), plain and fused.
TEST_F(NTileFlatParallelGrouped, MatchesAutoWithScaleGroups) {
    for (const type_case &t : kTypes) {
        if (t.wei != data_type_t::s8 && t.wei != data_type_t::s4) continue;
        for (const bool src_groups : {false, true}) {
            for (const bool fused : {false, true}) {
                SCOPED_TRACE(std::string(t.name)
                        + (src_groups ? " source groups" : " source rows")
                        + (fused ? " fused" : ""));
                grouped_case c = case_of(t);
                c.s8_groups = true;
                c.src_groups = src_groups;
                c.group = 32;
                c.fused = fused;
                c.bias = true;
                ASSERT_NO_FATAL_FAILURE(check_against_auto(c));
            }
        }
    }
}

// Gate and up values past SwiGLU-OAI's clamp limit.
TEST_F(NTileFlatParallelGrouped, MatchesAutoWhereTheActivationClamps) {
    for (const type_case &t : kTypes) {
        for (const bool fused : {false, true}) {
            SCOPED_TRACE(std::string(t.name) + (fused ? " fused" : ""));
            grouped_case c = case_of(t);
            c.act = act_t::swiglu_oai_mul;
            c.K = 64;
            c.N = 64;
            c.N_down = 32;
            c.group = 32;
            c.M = {32, 32, 32, 32};
            c.magnitude = 2.0f;
            c.fused = fused;
            ASSERT_NO_FATAL_FAILURE(check_against_auto(c));
        }
    }
}

// Up halves that are not a multiple of 16 columns, and widths that are not a
// multiple of the 32-column blocks.
TEST_F(NTileFlatParallelGrouped, MatchesAutoOnUnalignedWidths) {
    for (const type_case &t : kTypes) {
        if (t.wei == data_type_t::s4) continue;
        for (const int N : {200, 104}) {
            SCOPED_TRACE(std::string(t.name) + " N=" + std::to_string(N));
            grouped_case c = case_of(t);
            c.N = N;
            c.N_down = 72;
            c.fused = true;
            ASSERT_NO_FATAL_FAILURE(check_against_auto(c));
        }
    }
}

// A tight Op1 output (stride N / 2) holds what a wide one holds in its first
// N / 2 columns.
TEST_F(NTileFlatParallelGrouped, TightOutputMatchesWide) {
    AlgoEnvGuard pin(4);
    for (const type_case &t : kTypes) {
        if (t.wei == data_type_t::f16 && !fp16_host()) continue;
        SCOPED_TRACE(t.name);
        grouped_case c = case_of(t);
        const grouped_inputs in = make_inputs(c);
        reset_grp_matmul_caches();
        const grouped_result wide = run(c, in);
        c.ldc = c.N / 2;
        const grouped_result tight = run(c, in);
        ASSERT_EQ(wide.status, status_t::success);
        ASSERT_EQ(tight.status, status_t::success);
        EXPECT_EQ(wide.path, "ntile_flat_parallel");
        EXPECT_EQ(tight.path, "ntile_flat_parallel");
        EXPECT_EQ(tight.values, wide.values);
    }
    reset_grp_matmul_caches();
}

// Each phase request sends that phase's calls to N-tile flat parallel and
// leaves the other phase's, and calls of one operation, on AUTO.
TEST_F(NTileFlatParallelGrouped, RunsOnlyTheRequestedPhase) {
    grouped_case decode;
    decode.wei = data_type_t::s8;
    decode.M = {2, 1, 4, 3};
    grouped_case prompt = decode;
    prompt.M = {40, 70, 0, 33};
    grouped_case single = decode;
    single.M = {3};
    const grouped_inputs din = make_inputs(decode), pin = make_inputs(prompt),
                         sin = make_inputs(single);
    AlgoEnvGuard global(0);
    reset_grp_matmul_caches();
    {
        AutoDecodeAlgoOverride phase(4);
        EXPECT_EQ(run(decode, din).path, "ntile_flat_parallel");
        const grouped_result r = run(prompt, pin);
        EXPECT_EQ(r.status, status_t::success);
        EXPECT_NE(r.path, "ntile_flat_parallel");
    }
    {
        AutoPromptAlgoOverride phase(4);
        const grouped_result r = run(decode, din);
        EXPECT_EQ(r.status, status_t::success);
        EXPECT_NE(r.path, "ntile_flat_parallel");
        EXPECT_EQ(run(prompt, pin).path, "ntile_flat_parallel");
    }
    {
        AlgoEnvGuard pin4(4);
        const grouped_result r = run(single, sin);
        EXPECT_EQ(r.status, status_t::success);
        EXPECT_NE(r.path, "ntile_flat_parallel");
    }
    reset_grp_matmul_caches();
}

// Calls N-tile flat parallel cannot serve fail under it and leave their outputs
// as they were, next to a call it serves.
TEST_F(NTileFlatParallelGrouped, RefusesWhatItCannotServeWithoutWriting) {
    AlgoEnvGuard pin(4);
    reset_grp_matmul_caches();
    mm::matmul_post_op relu;
    relu.po_type = zendnnl::common::post_op_type_t::relu;
    int32_t zero_point = 1;
    float output_scale = 2.0f;
    using edit_t = std::function<void(call_args &)>;
    const auto second = [](void (*edit)(call_args &, int)) {
        return edit_t([edit](call_args &a) { edit(a, 1); });
    };
    grouped_case bf16;
    bf16.M = {2, 3};
    grouped_case s8 = bf16;
    s8.wei = data_type_t::s8;
    grouped_case post_op = s8;
    post_op.post_ops = {relu};
    grouped_case weight_only = bf16;
    weight_only.wei = data_type_t::s4;
    weight_only.dynamic_quant = false;
    grouped_case s4_rows = weight_only;
    s4_rows.src_s8 = true;
    grouped_case uncached = s8;
    uncached.weight_cache = 0;
    grouped_case biased = s8;
    biased.bias = true;
    grouped_case in_source = s8;
    in_source.fused = true;
    in_source.op1_internal = true;
    in_source.op2_internal = true;
    grouped_case source_groups = s8;
    source_groups.s8_groups = true;
    source_groups.src_groups = true;
    source_groups.group = 32;
    grouped_case fused = s8;
    fused.fused = true;
    // Upstream checks refuse a gated call with a column-major source first.
    grouped_case plain = s8;
    plain.act = act_t::none;
    struct unserved {
        const char *name;
        grouped_case c;
        edit_t edit;
    };
    const unserved cases[] = {
            {"control", s8, {}},
            {"post-op", post_op, {}},
            {"weight-only s4", weight_only, {}},
            {"s4 weights with the caller's s8 rows", s4_rows, {}},
            {"weight cache mode 0", uncached, {}},
            {"alpha", s8,
                    second([](call_args &a, int i) { a.alpha[i] = 2.0f; })},
            {"beta", s8, second([](call_args &a, int i) { a.beta[i] = 1.0f; })},
            {"transposed source", s8,
                    second([](call_args &a, int i) { a.transA[i] = true; })},
            {"column-major", plain,
                    second([](call_args &a, int i) { a.layout[i] = 'c'; })},
            {"non-constant weights", s8,
                    second([](call_args &a, int i) { a.wconst[i] = false; })},
            {"weight zero point", s8,
                    [&zero_point](call_args &a) {
        auto &zp = a.params[1].quant_params.wei_zp;
        zp.buff = &zero_point;
        zp.dt = data_type_t::s32;
        zp.dims = {1};
    }},
            {"destination scale", s8,
                    [&output_scale](call_args &a) {
        auto &scale = a.params[1].quant_params.dst_scale;
        scale.buff = &output_scale;
        scale.dt = data_type_t::f32;
        scale.dims = {1};
    }},
            {"prepacked weights", s8, second([](call_args &a, int i) {
        a.params[i].mem_format_b = 'r';
    })},
            {"GGML weights", s8, second([](call_args &a, int i) {
        a.params[i].packing.pack_format_b = 1;
    })},
            {"operations differ in K", s8,
                    second([](call_args &a, int i) { a.K[i] -= 8; })},
            {"operations differ in N", s8,
                    second([](call_args &a, int i) { a.N[i] -= 32; })},
            {"operations differ in the weight stride", s8,
                    second([](call_args &a, int i) { a.ldb[i] += 32; })},
            {"operations differ in the weight type", s8,
                    second([](call_args &a, int i) {
        a.params[i].dtypes.wei = data_type_t::s4;
    })},
            {"f32 output", bf16,
                    [](call_args &a) {
        for (mm::matmul_params &p : a.params)
            p.dtypes.dst = data_type_t::f32;
    }},
            {"s8 bias", biased,
                    [](call_args &a) {
        for (mm::matmul_params &p : a.params)
            p.dtypes.bias = data_type_t::s8;
    }},
            {"source rows as output with alpha", in_source,
                    second([](call_args &a, int i) { a.alpha[i] = 2.0f; })},
            {"source scales per row and per group", source_groups,
                    second([](call_args &a, int i) {
        a.params[i].quant_params.src_scale.dims[1] = 1;
    })},
            {"Op1 rows narrower than its output", fused,
                    [](call_args &a) {
        for (int &ld : a.ldc)
            ld = a.N[0] / 2 - 16;
    }},
    };
    for (const unserved &u : cases) {
        SCOPED_TRACE(u.name);
        const bool control = u.name == std::string("control");
        const grouped_inputs in = make_inputs(u.c);
        const grouped_result r = run(u.c, in, {}, -1, false, u.edit);
        ASSERT_FALSE(r.values.empty());
        if (control) {
            EXPECT_EQ(r.status, status_t::success);
            EXPECT_EQ(r.path, "ntile_flat_parallel");
            continue;
        }
        // N-tile flat parallel's own refusal, not an upstream check's.
        EXPECT_EQ(r.status, status_t::unimplemented);
        if (u.c.op2_internal) {
            // Op2 writes into the source rows: they hold the source still.
            std::vector<float> source;
            for (size_t i = 0; i < u.c.M.size(); ++i) {
                for (int j = 0; j < u.c.M[i]; ++j) {
                    const auto *row = reinterpret_cast<const uint16_t *>(
                                              in.src[i].data())
                            + static_cast<size_t>(j) * in.lda;
                    for (int d = 0; d < u.c.N_down; ++d)
                        source.push_back(from_bf16(row[d]));
                }
            }
            EXPECT_EQ(r.values, source) << "the refused call wrote its output";
            continue;
        }
        EXPECT_TRUE(std::all_of(
                r.values.begin(), r.values.end(), [](const float v) {
            return std::isnan(v);
        })) << "the refused call wrote its output";
        EXPECT_TRUE(std::all_of(r.op1.begin(), r.op1.end(), [](const float v) {
            return std::isnan(v);
        })) << "the refused call wrote Op1's rows";
        for (const float v : r.src_scales) {
            EXPECT_EQ(v, 0.0f) << "the refused call wrote its source scales";
        }
    }
    reset_grp_matmul_caches();
}

// The first call packs its experts and the prepack extras; a call of other
// experts of the same weights packs nothing; the flush drops every pack.
TEST_F(NTileFlatParallelGrouped, PacksTheExtrasOnceAndFlushes) {
    grouped_case c;
    c.wei = data_type_t::s8;
    c.fused = true;
    c.M = {2, 3, 4, 1};
    const grouped_inputs in = make_inputs(c);
    AlgoEnvGuard pin(4);
    reset_grp_matmul_caches();
    ASSERT_EQ(mm::test_api::ntile_flat_parallel_packs(), 0u);
    const grouped_result first = run(c, in, {0, 1, 2, 3}, /*active=*/2);
    ASSERT_EQ(first.status, status_t::success);
    EXPECT_EQ(first.path, "ntile_flat_parallel");
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 8u);
    const grouped_result other = run(c, in, {3, 2, 1, 0}, /*active=*/2);
    ASSERT_EQ(other.status, status_t::success);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 8u);
    // The packs a call made agree with the ones made for the whole set.
    reset_grp_matmul_caches();
    EXPECT_EQ(run(c, in, {3, 2, 1, 0}, /*active=*/2).values, other.values);
    mm::clear_grp_matmul_weight_caches();
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 0u);
    reset_grp_matmul_caches();
}

// Calls that name the same weights, in any order and with other experts
// given rows, share one call site, and write what calls through the store
// alone write.
TEST_F(NTileFlatParallelGrouped, CallsNamingTheSameWeightsShareOneSite) {
    for (const data_type_t wei : {data_type_t::bf16, data_type_t::s8}) {
        SCOPED_TRACE(wei == data_type_t::s8 ? "da8w8" : "bf16");
        grouped_case c;
        c.wei = wei;
        c.fused = true;
        c.M = {2, 3, 4, 1};
        const grouped_inputs in = make_inputs(c);
        AlgoEnvGuard pin(4);
        reset_grp_matmul_caches();
        const grouped_result first = run(c, in, {0, 1, 2, 3}, /*active=*/2);
        ASSERT_EQ(first.status, status_t::success);
        EXPECT_EQ(mm::test_api::ntile_flat_parallel_call_sites(), 1u);
        EXPECT_EQ(run(c, in, {0, 1, 2, 3}, 2).values, first.values);
        const grouped_result other = run(c, in, {3, 2, 1, 0}, 2);
        ASSERT_EQ(other.status, status_t::success);
        EXPECT_EQ(mm::test_api::ntile_flat_parallel_call_sites(), 1u);
        reset_grp_matmul_caches();
        EXPECT_EQ(run(c, in, {3, 2, 1, 0}, 2).values, other.values);
        mm::clear_grp_matmul_weight_caches();
        EXPECT_EQ(mm::test_api::ntile_flat_parallel_call_sites(), 0u);
    }
    reset_grp_matmul_caches();
}

// Extras named without their weight scales, as frameworks pass them, are
// packed on the first call all the same; an extra that later gets rows
// packs nothing more, and both calls write what AUTO writes.
TEST_F(NTileFlatParallelGrouped, PacksExtrasThatComeWithoutTheirScales) {
    for (const data_type_t wei : {data_type_t::s8, data_type_t::s4}) {
        SCOPED_TRACE(wei == data_type_t::s8 ? "da8w8" : "w4a8");
        grouped_case c;
        c.wei = wei;
        c.fused = true;
        c.M = {2, 3, 4, 1};
        const grouped_inputs in = make_inputs(c);
        reset_grp_matmul_caches();
        std::vector<grouped_result> requested;
        {
            AlgoEnvGuard pin(4);
            requested.push_back(run(c, in, {0, 1, 2, 3}, 2, true));
            ASSERT_EQ(requested.back().status, status_t::success);
            EXPECT_EQ(requested.back().path, "ntile_flat_parallel");
            EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 8u);
            requested.push_back(run(c, in, {2, 3, 0, 1}, 2, true));
            ASSERT_EQ(requested.back().status, status_t::success);
            EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 8u);
        }
        reset_grp_matmul_caches();
        AlgoEnvGuard pin(0);
        expect_close(requested[0], run(c, in, {0, 1, 2, 3}, 2, true), 0.03);
        expect_close(requested[1], run(c, in, {2, 3, 0, 1}, 2, true), 0.03);
        reset_grp_matmul_caches();
    }
}

/// Every value of `r` is NaN, Op1's rows in the caller's `dst` included: the
/// call wrote none of its rows.
bool unwritten(const grouped_result &r) {
    const auto nan = [](const float v) { return std::isnan(v); };
    return !r.values.empty()
            && std::all_of(r.values.begin(), r.values.end(), nan)
            && std::all_of(r.op1.begin(), r.op1.end(), nan);
}

// DA8W8 calls write what AUTO writes with either source kernel, and run the
// one pinned.
TEST_F(NTileFlatParallelGrouped, MatchesAutoWithEitherDa8w8SourceKernel) {
    auto &u8_calls = mm::test_api::s_ntile_flat_parallel_u8_src_calls;
    for (const bool src_s8 : {false, true}) {
        for (const int u8 : {0, 1}) {
            for (const bool fused : {false, true}) {
                SCOPED_TRACE(std::string(src_s8 ? "s8 rows" : "bf16 rows")
                        + (u8 ? " u8 kernels" : " s8 kernels")
                        + (fused ? " fused" : ""));
                hook_pin kernels(
                        mm::test_api::s_ntile_flat_parallel_u8_src_override,
                        u8);
                hook_pin capture(
                        mm::test_api::s_capture_ntile_flat_parallel, true);
                const uint64_t before = u8_calls.load();
                grouped_case c;
                c.wei = data_type_t::s8;
                c.src_s8 = src_s8;
                c.fused = fused;
                c.bias = true;
                ASSERT_NO_FATAL_FAILURE(check_against_auto(c));
                EXPECT_EQ(u8_calls.load() - before, u8 ? 1u : 0u);
            }
        }
    }
}

// Both decode schedules, at every team size, write the same bytes.
TEST_F(NTileFlatParallelGrouped, DecodeSchedulesAndTeamSizesWriteTheSameBytes) {
    AlgoEnvGuard pin(4);
    for (const type_case &t : kTypes) {
        if (t.wei == data_type_t::f16 && !fp16_host()) continue;
        SCOPED_TRACE(t.name);
        grouped_case c = case_of(t);
        c.fused = true;
        const grouped_inputs in = make_inputs(c);
        reset_grp_matmul_caches();
        std::vector<float> first;
        for (const int threads : {1, 3, 7}) {
            for (const int own : {0, 1}) {
                SCOPED_TRACE(std::to_string(threads) + " threads"
                        + (own ? " own units first" : " queue order"));
                hook_pin schedule(
                        mm::test_api::s_ntile_flat_parallel_own_units_override,
                        own);
                c.threads = threads;
                const grouped_result r = run(c, in);
                ASSERT_EQ(r.status, status_t::success);
                EXPECT_EQ(r.path, "ntile_flat_parallel");
                if (first.empty()) {
                    first = r.values;
                } else {
                    EXPECT_EQ(r.values, first);
                }
            }
        }
    }
    reset_grp_matmul_caches();
}

/// Reads `text` as N-tile flat parallel's config and drops every plan made
/// under the last.
void use_ntile_flat_parallel_config(const char *text) {
    mm::test_api::set_ntile_flat_parallel_config(text);
    mm::ntile_flat_parallel::flush();
    reset_grp_matmul_caches();
}

/// Reads the environment's config again when it goes, as a test that
/// `ASSERT`s out of its loop still must.
struct config_restore {
    ~config_restore() { use_ntile_flat_parallel_config(nullptr); }
};

// Every N-tile flat parallel setting changes only how a call's work is cut,
// ordered and staged, so each writes the same bytes as the defaults, in both
// phases.
TEST_F(NTileFlatParallelGrouped, SettingsWriteTheSameBytes) {
    AlgoEnvGuard pin(4);
    const config_restore restore;
    const char *const configs[] = {
            "units=1,prompt_units=1",
            "units=8,prompt_units=12",
            "sweep_rows=1",
            "sweep_rows=100000",
            "unit_order=queue",
            "unit_order=own",
            "heavy_light=off",
            "l2_frac=0.25,subtile_rows=4",
            "l2_frac=1",
            "u8_src=on",
            "u8_src=off",
            "wait_spins=0",
    };
    for (const type_case &t : kTypes) {
        if (t.wei == data_type_t::f16 && !fp16_host()) continue;
        for (const bool prompt : {false, true}) {
            SCOPED_TRACE(
                    std::string(t.name) + (prompt ? " prompt" : " decode"));
            grouped_case c = case_of(t);
            c.fused = true;
            if (prompt) c.M = {70, 3, 0, 129};
            const grouped_inputs in = make_inputs(c);
            use_ntile_flat_parallel_config("");
            const grouped_result first = run(c, in);
            ASSERT_EQ(first.status, status_t::success);
            for (const char *config : configs) {
                SCOPED_TRACE(config);
                use_ntile_flat_parallel_config(config);
                EXPECT_TRUE(mm::test_api::ntile_flat_parallel_config_warnings()
                                    .empty());
                const grouped_result r = run(c, in);
                ASSERT_EQ(r.status, status_t::success);
                EXPECT_EQ(r.path, "ntile_flat_parallel");
                EXPECT_EQ(r.values, first.values);
            }
        }
    }
}

// The call settings reach the unit queue: units per thread in decode and in
// prompt, which follow the call's phase whatever sweeps, and the rows above
// which units sweep.
TEST_F(NTileFlatParallelGrouped, CallSettingsReachTheQueue) {
    AlgoEnvGuard pin(4);
    const config_restore restore;
    hook_pin capture(mm::test_api::s_capture_ntile_flat_parallel, true);
    const auto queue_of = [](const char *config, const grouped_case &c) {
        use_ntile_flat_parallel_config(config);
        const grouped_result r = run(c, make_inputs(c));
        EXPECT_EQ(r.status, status_t::success);
        return std::make_pair(
                mm::test_api::s_ntile_flat_parallel_last_units.load(),
                mm::test_api::s_ntile_flat_parallel_last_sweep.load());
    };
    grouped_case decode;
    decode.wei = data_type_t::bf16;
    decode.threads = 8;
    grouped_case prompt = decode;
    prompt.M = {70, 3, 0, 129};
    const auto d1 = queue_of("units=1", decode);
    const auto d8 = queue_of("units=8", decode);
    EXPECT_FALSE(d1.second);
    EXPECT_LT(d1.first, d8.first);
    EXPECT_TRUE(queue_of("sweep_rows=1", decode).second);
    const auto p1 = queue_of("prompt_units=1", prompt);
    const auto p12 = queue_of("prompt_units=12", prompt);
    EXPECT_TRUE(p1.second);
    EXPECT_LT(p1.first, p12.first);
    EXPECT_FALSE(queue_of("sweep_rows=1000", prompt).second);
    EXPECT_LT(queue_of("sweep_rows=1,units=1,prompt_units=12", decode).first,
            queue_of("sweep_rows=1,units=8,prompt_units=12", decode).first);
    EXPECT_LT(queue_of("sweep_rows=1000,units=12,prompt_units=1", prompt).first,
            queue_of("sweep_rows=1000,units=12,prompt_units=12", prompt).first);
}

// Where no test pin does, the u8_src setting picks DA8W8's source kernels.
TEST_F(NTileFlatParallelGrouped, TheU8SourceSettingPicksTheKernels) {
    AlgoEnvGuard pin(4);
    const config_restore restore;
    auto &u8_calls = mm::test_api::s_ntile_flat_parallel_u8_src_calls;
    for (const bool on : {false, true}) {
        SCOPED_TRACE(on ? "u8_src=on" : "u8_src=off");
        use_ntile_flat_parallel_config(on ? "u8_src=on" : "u8_src=off");
        hook_pin capture(mm::test_api::s_capture_ntile_flat_parallel, true);
        const uint64_t before = u8_calls.load();
        grouped_case c;
        c.wei = data_type_t::s8;
        const grouped_result r = run(c, make_inputs(c));
        ASSERT_EQ(r.status, status_t::success);
        EXPECT_EQ(u8_calls.load() - before, on ? 1u : 0u);
    }
}

// New scale buffers for weights already packed give the call their values;
// the weights' packs stay.
TEST_F(NTileFlatParallelGrouped, MovedScalesKeepTheWeightPacks) {
    for (const type_case &t : kTypes) {
        if (!quantized(case_of(t))) continue;
        SCOPED_TRACE(t.name);
        grouped_case c = case_of(t);
        c.fused = true;
        const grouped_inputs in = make_inputs(c);
        std::vector<std::vector<float>> up, down;
        for (const auto &w : in.w13) {
            up.push_back(w.scale);
            for (float &v : up.back())
                v *= 1.25f;
        }
        for (const auto &w : in.w2) {
            down.push_back(w.scale);
            for (float &v : down.back())
                v *= 0.8f;
        }
        const auto moved = [&](call_args &a) {
            for (size_t i = 0; i < a.params.size(); ++i)
                a.params[i].quant_params.wei_scale.buff = up[i].data();
            for (size_t i = 0; i < a.fused.down_scale.size(); ++i)
                a.fused.down_scale[i].buff = down[i].data();
        };
        reset_grp_matmul_caches();
        grouped_result before, after;
        {
            AlgoEnvGuard pin(4);
            before = run(c, in);
            ASSERT_EQ(before.status, status_t::success);
            const size_t packs = mm::test_api::ntile_flat_parallel_packs();
            after = run(c, in, {}, -1, false, moved);
            ASSERT_EQ(after.status, status_t::success);
            EXPECT_EQ(after.path, "ntile_flat_parallel");
            EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), packs);
        }
        EXPECT_NE(after.values, before.values);
        reset_grp_matmul_caches();
        AlgoEnvGuard pin(0);
        expect_close(after, run(c, in, {}, -1, false, moved), 0.03);
        reset_grp_matmul_caches();
    }
}

// A call site serves an expert only with the scales its pack holds: a call
// that moves one projection's scales, the other's staying, writes what it
// writes on fresh packs.
TEST_F(NTileFlatParallelGrouped, ACallSiteServesOnlyTheScalesItWasMadeWith) {
    for (const type_case &t : kTypes) {
        if (!quantized(case_of(t))) continue;
        // 0: a call without a down projection moves W13's scales; 1 and 2: a
        // fused call moves W13's and W2's alone.
        for (const int moves : {0, 1, 2}) {
            SCOPED_TRACE(std::string(t.name) + " moves "
                    + (moves == 0                ? "W13, not fused"
                                    : moves == 1 ? "W13"
                                                 : "W2"));
            grouped_case c = case_of(t);
            c.fused = moves > 0;
            const grouped_inputs in = make_inputs(c);
            std::vector<std::vector<float>> up, down;
            for (const auto &w : in.w13) {
                up.push_back(w.scale);
                for (float &v : up.back())
                    v *= 1.25f;
            }
            for (const auto &w : in.w2) {
                down.push_back(w.scale);
                for (float &v : down.back())
                    v *= 0.8f;
            }
            const auto moved = [&](call_args &a) {
                for (size_t i = 0; moves < 2 && i < a.params.size(); ++i)
                    a.params[i].quant_params.wei_scale.buff = up[i].data();
                for (size_t i = 0; moves == 2 && i < a.fused.down_scale.size();
                        ++i)
                    a.fused.down_scale[i].buff = down[i].data();
            };
            AlgoEnvGuard pin(4);
            reset_grp_matmul_caches();
            const grouped_result before = run(c, in);
            ASSERT_EQ(before.status, status_t::success);
            const grouped_result after = run(c, in, {}, -1, false, moved);
            ASSERT_EQ(after.status, status_t::success);
            EXPECT_EQ(after.path, "ntile_flat_parallel");
            EXPECT_NE(after.values, before.values);
            reset_grp_matmul_caches();
            const grouped_result fresh = run(c, in, {}, -1, false, moved);
            ASSERT_EQ(fresh.status, status_t::success);
            EXPECT_EQ(after.values, fresh.values)
                    << "the site served the old scales";
        }
    }
    reset_grp_matmul_caches();
}

// Calls that name only their experts with rows, each pair in turn, with the
// scales in other buffers every time: the store keeps one scaled pack per
// weight, and no call site keeps one the store replaced.
TEST_F(NTileFlatParallelGrouped, MovedScalesLeaveOneScaledPackPerWeight) {
    grouped_case c;
    c.wei = data_type_t::s8;
    c.fused = true;
    c.M = {2, 3, 4, 1};
    const int E = static_cast<int>(c.M.size());
    const grouped_inputs in = make_inputs(c);
    // Three buffers of each expert's scales.
    std::vector<std::vector<std::vector<float>>> up(3), down(3);
    for (int k = 0; k < 3; ++k) {
        for (int e = 0; e < E; ++e) {
            up[k].push_back(in.w13[e].scale);
            down[k].push_back(in.w2[e].scale);
        }
    }
    const auto &live = mm::test_api::s_ntile_flat_parallel_live_packs;
    AlgoEnvGuard pin(4);
    reset_grp_matmul_caches();
    ASSERT_EQ(live.load(), 0);
    for (int i = 0; i < 4 * E; ++i) {
        const std::vector<int> order {i % E, (i + 1) % E};
        const int k = i % 3;
        const grouped_result r
                = run(c, in, order, -1, false, [&](call_args &a) {
            for (size_t j = 0; j < order.size(); ++j) {
                a.params[j].quant_params.wei_scale.buff
                        = up[k][order[j]].data();
                a.fused.down_scale[j].buff = down[k][order[j]].data();
            }
        });
        ASSERT_EQ(r.status, status_t::success);
        EXPECT_EQ(r.path, "ntile_flat_parallel");
    }
    // A bare and a scaled pack of each W13 and W2 weight.
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(),
            static_cast<size_t>(2 * E));
    EXPECT_EQ(live.load(), 2 * 2 * E);
    reset_grp_matmul_caches();
    EXPECT_EQ(live.load(), 0);
}

// At most 1,024 call sites and 4,096 pools whose prepack extras were packed
// are kept: one more drops them all.
TEST_F(NTileFlatParallelGrouped, CallSitesAndPoolFingerprintsAreBounded) {
    grouped_case c;
    c.act = act_t::none;
    c.K = 32;
    c.N = 32;
    c.M.assign(48, 1);
    const int E = static_cast<int>(c.M.size());
    const grouped_inputs in = make_inputs(c);
    AlgoEnvGuard pin(4);
    reset_grp_matmul_caches();
    // Each pair of experts is a site of its own.
    size_t calls = 0;
    for (int a = 0; a < E && calls <= 1024; ++a) {
        for (int b = a + 1; b < E && calls <= 1024; ++b) {
            ASSERT_EQ(run(c, in, {a, b}).status, status_t::success);
            ++calls;
            if (calls == 1024) {
                ASSERT_EQ(
                        mm::test_api::ntile_flat_parallel_call_sites(), 1024u);
            }
        }
    }
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_call_sites(), 1u);
    reset_grp_matmul_caches();
    // Each pair with a third expert as its extra is a pool of its own.
    calls = 0;
    for (int a = 0; a < E && calls <= 4096; ++a) {
        for (int b = a + 1; b < E && calls <= 4096; ++b) {
            for (int x = b + 1; x < E && calls <= 4096; ++x) {
                ASSERT_EQ(run(c, in, {a, b, x}, 2).status, status_t::success);
                ++calls;
                if (calls == 4096) {
                    ASSERT_EQ(mm::test_api::ntile_flat_parallel_warmed_pools(),
                            4096u);
                }
            }
        }
    }
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_warmed_pools(), 1u);
    reset_grp_matmul_caches();
}

// A call site serves only calls of the tile plans it holds: the same weights
// with another activation, other source kernels or source scales of other
// groups make a second site, and the second call writes what it writes on
// fresh packs.
TEST_F(NTileFlatParallelGrouped, ACallSiteServesOnlyTheCallsOfItsPlans) {
    auto &u8 = mm::test_api::s_ntile_flat_parallel_u8_src_override;
    for (const int change : {0, 1, 2}) {
        SCOPED_TRACE(change == 0      ? "activation"
                        : change == 1 ? "source kernels"
                                      : "source scale groups");
        grouped_case first;
        first.wei = data_type_t::s8;
        first.s8_groups = change == 2;
        grouped_case second = first;
        int u8_first = 0, u8_second = 0;
        if (change == 0) second.act = act_t::gelu_and_mul;
        if (change == 1) u8_second = 1;
        if (change == 2) second.src_groups = true;
        const grouped_inputs in = make_inputs(second);
        AlgoEnvGuard pin(4);
        reset_grp_matmul_caches();
        {
            hook_pin kernels(u8, u8_first);
            ASSERT_EQ(run(first, in).status, status_t::success);
        }
        grouped_result again, fresh;
        {
            hook_pin kernels(u8, u8_second);
            again = run(second, in);
            ASSERT_EQ(again.status, status_t::success);
            EXPECT_EQ(mm::test_api::ntile_flat_parallel_call_sites(), 2u);
            reset_grp_matmul_caches();
            fresh = run(second, in);
            ASSERT_EQ(fresh.status, status_t::success);
        }
        EXPECT_EQ(again.values, fresh.values)
                << "the second call ran on the first call's plans";
    }
    reset_grp_matmul_caches();
}

// One set of weight buffers read at two shapes, [K, N] and then [N, K] with K
// and N swapped, makes two call sites: the second call writes what it writes
// on fresh packs.
TEST_F(NTileFlatParallelGrouped, OneSetOfBuffersAtTwoShapesMakesTwoCallSites) {
    grouped_case wide;
    wide.K = 192;
    wide.N = 256;
    grouped_case tall = wide;
    tall.K = 256;
    tall.N = 192;
    tall.trans = true;
    const grouped_inputs in = make_inputs(wide);
    const grouped_inputs in_tall = make_inputs(tall);
    const auto same_buffers = [&](call_args &a) {
        for (size_t i = 0; i < a.weight.size(); ++i)
            a.weight[i] = in.w13[i].data.data();
    };
    AlgoEnvGuard pin(4);
    reset_grp_matmul_caches();
    ASSERT_EQ(run(wide, in).status, status_t::success);
    const grouped_result second
            = run(tall, in_tall, {}, -1, false, same_buffers);
    ASSERT_EQ(second.status, status_t::success);
    EXPECT_EQ(second.path, "ntile_flat_parallel");
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_call_sites(), 2u);
    reset_grp_matmul_caches();
    const grouped_result fresh
            = run(tall, in_tall, {}, -1, false, same_buffers);
    ASSERT_EQ(fresh.status, status_t::success);
    EXPECT_EQ(second.values, fresh.values)
            << "the second shape ran on the first one's packs";
    reset_grp_matmul_caches();
}

// A pack that runs out of memory fails its call with a memory status and
// writes nothing; the next call packs and runs.
TEST_F(NTileFlatParallelGrouped,
        RunningOutOfMemoryWhilePackingFailsOnlyThatCall) {
    grouped_case c;
    c.wei = data_type_t::s8;
    c.fused = true;
    const grouped_inputs in = make_inputs(c);
    AlgoEnvGuard pin(4);
    reset_grp_matmul_caches();
    {
        hook_pin fails(mm::ntile_flat_parallel::nr32::test_api::
                               s_pack_allocation_fails,
                true);
        const grouped_result r = run(c, in);
        EXPECT_EQ(r.status, status_t::memory_bad_storage);
        ASSERT_FALSE(r.op1.empty());
        EXPECT_TRUE(unwritten(r));
        EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 0u);
    }
    const grouped_result r = run(c, in);
    EXPECT_EQ(r.status, status_t::success);
    EXPECT_EQ(r.path, "ntile_flat_parallel");
    reset_grp_matmul_caches();
}

// At weight cache mode 2, while N-tile flat parallel is requested for a phase,
// the calls it does not serve (one operation of an MoE layer, or the other
// phase) leave the caller's weights raw, and its own calls are served.
TEST_F(NTileFlatParallelGrouped, OtherCallsLeaveTheWeightsRawAtCacheMode2) {
    config_.set_weight_cache(2);
    struct setting {
        const char *name;
        int algo;
        int decode;
        bool prompt_served;
    };
    const setting settings[] = {
            {"ALGO=4", 4, 7, true},
            {"AUTO_DECODE_ALGO=4", 0, 4, false},
            {"ALGO=3 and AUTO_DECODE_ALGO=4", 3, 4, false},
    };
    for (const setting &st : settings) {
        SCOPED_TRACE(st.name);
        AlgoEnvGuard algo(st.algo);
        AutoDecodeAlgoOverride decode(st.decode);
        grouped_case c;
        c.fused = true;
        c.weight_cache = 2;
        c.M = {2, 1, 4, 3};
        grouped_case prompt = c;
        prompt.M = {40, 70, 33, 50};
        grouped_case one = c;
        one.M = {3};
        grouped_case one_prompt = c;
        one_prompt.M = {300};
        // One operation whose only MoE part is the weighted reduce.
        grouped_case one_reduce = c;
        one_reduce.fused = false;
        one_reduce.act = act_t::none;
        one_reduce.M.assign(1, 0);
        one_reduce.tokens = 3;
        one_reduce.topk = 1;
        const grouped_inputs in = make_inputs(c),
                             prompt_in = make_inputs(prompt, 11),
                             one_in = make_inputs(one, 13),
                             one_prompt_in = make_inputs(one_prompt, 17),
                             one_reduce_in = make_inputs(one_reduce, 19);
        const auto weights = [](const grouped_inputs &i) {
            std::vector<std::vector<uint8_t>> w;
            for (const auto &x : i.w13)
                w.push_back(x.data);
            for (const auto &x : i.w2)
                w.push_back(x.data);
            return w;
        };
        const auto raw = weights(in), prompt_raw = weights(prompt_in),
                   one_raw = weights(one_in),
                   one_prompt_raw = weights(one_prompt_in),
                   one_reduce_raw = weights(one_reduce_in);
        reset_grp_matmul_caches();

        grouped_result r = run(one, one_in);
        EXPECT_EQ(r.status, status_t::success);
        EXPECT_NE(r.path, "ntile_flat_parallel");
        r = run(one_prompt, one_prompt_in);
        EXPECT_EQ(r.status, status_t::success);
        r = run(one_reduce, one_reduce_in);
        EXPECT_EQ(r.status, status_t::success);
        r = run(c, in, {0, 1, 2, 3}, /*active=*/1);
        EXPECT_EQ(r.status, status_t::success);
        EXPECT_NE(r.path, "ntile_flat_parallel");
        r = run(prompt, prompt_in);
        EXPECT_EQ(r.status, status_t::success);
        EXPECT_EQ(r.path == "ntile_flat_parallel", st.prompt_served);
        r = run(c, in);
        EXPECT_EQ(r.status, status_t::success);
        EXPECT_EQ(r.path, "ntile_flat_parallel");

        EXPECT_TRUE(weights(in) == raw) << "a call packed the weights in place";
        EXPECT_TRUE(weights(prompt_in) == prompt_raw);
        EXPECT_TRUE(weights(one_in) == one_raw);
        EXPECT_TRUE(weights(one_prompt_in) == one_prompt_raw);
        EXPECT_TRUE(weights(one_reduce_in) == one_reduce_raw);
        mm::clear_grp_matmul_weight_caches();
        reset_grp_matmul_caches();
    }
}

/// Rewrites expert weight `w` of `c` (bf16 `[K, N]`) in place as a dense
/// oneDNN call at weight cache mode 2 does.  With the custom kernel on, a
/// dense call keeps a raw weight for it instead.
void rewrite_with_onednn(const grouped_case &c, std::vector<uint8_t> &w) {
    hook_pin custom_kernel(
            mm::test_api::s_grp_matmul_custom_kernel_override, 0);
    const int M = 32;
    std::vector<uint16_t> src(static_cast<size_t>(M) * c.K, to_bf16(0.5f));
    std::vector<uint16_t> dst(static_cast<size_t>(M) * c.N);
    mm::matmul_params p;
    p.dtypes.src = data_type_t::bf16;
    p.dtypes.wei = data_type_t::bf16;
    p.dtypes.dst = data_type_t::bf16;
    p.lowoha_algo = zendnnl::common::matmul_algo_t::onednn_blocked;
    p.weight_cache_type = 2;
    mm::matmul_batch_params_t batch;
    ASSERT_EQ(mm::matmul_direct('r', false, false, M, c.N, c.K, 1.0f,
                      src.data(), c.K, w.data(), c.N, nullptr, 0.0f, dst.data(),
                      c.N, true, batch, p),
            status_t::success);
}

/// The same with the native BRGEMM's layout.
void rewrite_with_native(const grouped_case &c, std::vector<uint8_t> &w) {
    namespace nv = mm::native;
    nv::native_raw_weight_t r;
    r.wei_dt = data_type_t::bf16;
    r.weight = w.data();
    r.K = c.K;
    r.N = c.N;
    r.ldb = c.N;
    r.transB = false;
    r.pack_nr = 32;
    nv::native_layout_t layout;
    ASSERT_EQ(nv::get_native_layout(r, 2, 0, 4, layout), status_t::success);
    ASSERT_EQ(layout.data, static_cast<const void *>(w.data()));
}

// Weights another path rewrote into the caller's buffers no longer hold the
// operation's weights: N-tile flat parallel refuses them rather than read them,
// also after another backend's weight cache is cleared.
TEST_F(NTileFlatParallelGrouped, RefusesWeightsPackedInPlace) {
    enum class rewriter { aocl, custom_kernel, onednn, native };
    const std::pair<const char *, rewriter> rewriters[] = {
            {"AOCL-DLP", rewriter::aocl},
            {"custom kernel", rewriter::custom_kernel},
            {"oneDNN", rewriter::onednn},
            {"native BRGEMM", rewriter::native},
    };
    config_.set_weight_cache(2);
    for (const auto &[name, by] : rewriters) {
        SCOPED_TRACE(name);
        grouped_case c;
        c.fused = true;
        c.weight_cache = 2;
        c.M = {40, 70, 33, 50};
        // A shape whose oneDNN blocked layout fills the raw weight's bytes.
        c.K = 256;
        grouped_inputs in = make_inputs(c);
        const std::vector<uint8_t> raw = in.w13[0].data;
        reset_grp_matmul_caches();
        if (by == rewriter::onednn) {
            rewrite_with_onednn(c, in.w13[0].data);
        } else if (by == rewriter::native) {
            rewrite_with_native(c, in.w13[0].data);
        } else {
            AlgoEnvGuard algo(by == rewriter::aocl ? 1 : 3);
            hook_pin custom_kernel(
                    mm::test_api::s_grp_matmul_custom_kernel_override,
                    by == rewriter::aocl ? 0 : 1);
            ASSERT_EQ(run(c, in).status, status_t::success);
        }
        ASSERT_NE(in.w13[0].data, raw) << "the weight must be rewritten";
        if (by == rewriter::aocl) mm::clear_matmul_onednn_weight_caches();
        if (by == rewriter::onednn) mm::clear_matmul_aocl_weight_caches();
        AlgoEnvGuard pin(4);
        const grouped_result r = run(c, in);
        EXPECT_EQ(r.status, status_t::unimplemented);
        EXPECT_TRUE(unwritten(r));
        mm::clear_grp_matmul_weight_caches();
        reset_grp_matmul_caches();
    }
}

// FP16 calls on a host without AVX512-FP16 fail before anything is packed.
TEST_F(NTileFlatParallelGrouped, RefusesFp16WithoutItsKernelsBeforePacking) {
    if (fp16_host()) GTEST_SKIP() << "this host has the FP16 kernels";
    AlgoEnvGuard pin(4);
    reset_grp_matmul_caches();
    for (const bool fused : {false, true}) {
        SCOPED_TRACE(fused ? "fused" : "plain");
        grouped_case c;
        c.wei = data_type_t::f16;
        c.fused = fused;
        const grouped_result r = run(c, make_inputs(c));
        EXPECT_NE(r.status, status_t::success);
        EXPECT_TRUE(unwritten(r));
        EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 0u);
    }
}

// A W4A8 down projection whose depth is not whole octets, with one scale per
// column, has no kernel: the call fails before anything is packed.
TEST_F(NTileFlatParallelGrouped, RefusesW4a8ScalesNoKernelReadsBeforePacking) {
    AlgoEnvGuard pin(4);
    reset_grp_matmul_caches();
    grouped_case c;
    c.wei = data_type_t::s4;
    c.fused = true;
    c.N = 72;
    const grouped_inputs in = make_inputs(c);
    std::vector<float> per_column(static_cast<size_t>(c.N_down), 0.01f);
    const grouped_result r = run(c, in, {}, -1, false, [&](call_args &a) {
        for (auto &s : a.fused.down_scale) {
            s.buff = per_column.data();
            s.dims = {1, c.N_down};
        }
    });
    EXPECT_EQ(r.status, status_t::unimplemented);
    EXPECT_TRUE(unwritten(r));
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 0u);
}

// A fused call without its down widths or an output's strides fails rather
// than read past them, whether or not the public API's diagnostics check it
// first.  The emptied vectors own no storage, so a read past them faults.
TEST_F(NTileFlatParallelGrouped, RefusesAFusedCallWithoutItsWidthsOrStrides) {
    AlgoEnvGuard pin(4);
    reset_grp_matmul_caches();
    grouped_case c;
    c.wei = data_type_t::s8;
    c.fused = true;
    const grouped_inputs in = make_inputs(c);
    using edit_t = std::function<void(call_args &)>;
    const std::pair<const char *, edit_t> edits[] = {
            {"no down widths",
                    [](call_args &a) {
        std::vector<int>().swap(a.fused.N_down);
    }},
            {"no Op1 strides",
                    [](call_args &a) {
        a.dst[1] = nullptr;
        std::vector<int>().swap(a.ldc);
    }},
            {"no Op2 strides",
                    [](call_args &a) {
        std::vector<int>().swap(a.fused.ldc_down);
    }},
    };
    for (const auto &entry : edits) {
        const char *const name = entry.first;
        const edit_t &edit = entry.second;
        for (const bool direct : {false, true}) {
            SCOPED_TRACE(std::string(name)
                    + (direct ? ", grouped executor" : ", public API"));
            const grouped_result r
                    = run(c, in, {}, -1, false, [&edit, direct](call_args &a) {
                a.direct = direct;
                edit(a);
            });
            EXPECT_NE(r.status, status_t::success);
            EXPECT_TRUE(unwritten(r));
        }
    }
    reset_grp_matmul_caches();
}

// The caller's s8 rows with one source scale, in a buffer of that scale
// alone, write what they write with that scale given to every row.
TEST_F(NTileFlatParallelGrouped, OneSourceScaleForS8RowsMatchesThePerRowForm) {
    AlgoEnvGuard pin(4);
    for (const bool fused : {false, true}) {
        SCOPED_TRACE(fused ? "fused" : "plain");
        grouped_case c;
        c.wei = data_type_t::s8;
        c.src_s8 = true;
        c.fused = fused;
        grouped_inputs in = make_inputs(c);
        // Per operation: its scale, then NaN past it.
        std::vector<float> one(2 * c.M.size());
        for (size_t i = 0; i < c.M.size(); ++i) {
            auto &scales = in.src_scale[i];
            std::fill(scales.begin(), scales.end(), scales.front());
            one[2 * i] = scales.front();
            one[2 * i + 1] = std::numeric_limits<float>::quiet_NaN();
        }
        reset_grp_matmul_caches();
        const grouped_result rows = run(c, in);
        ASSERT_EQ(rows.status, status_t::success);
        const grouped_result single
                = run(c, in, {}, -1, false, [&one](call_args &a) {
            for (size_t i = 0; i < a.params.size(); ++i) {
                auto &q = a.params[i].quant_params.src_scale;
                q.buff = &one[2 * i];
                q.dims = {1};
            }
        });
        ASSERT_EQ(single.status, status_t::success);
        EXPECT_EQ(single.path, "ntile_flat_parallel");
        EXPECT_EQ(single.values, rows.values);
    }
    reset_grp_matmul_caches();
}

// A dense projection's gated call is no MoE call: while N-tile flat parallel is
// requested, it packs its weight in place at weight cache mode 2 as it does
// without N-tile flat parallel.
TEST_F(NTileFlatParallelGrouped,
        DenseGatedProjectionsPackAsWithoutNTileFlatParallel) {
    config_.set_weight_cache(2);
    const int M = 1, K = 512, N = 2048;
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::vector<uint16_t> src(static_cast<size_t>(M) * K),
            raw(static_cast<size_t>(N) * K);
    for (uint16_t &v : src)
        v = to_bf16(unit(rng));
    for (uint16_t &v : raw)
        v = to_bf16(0.05f * unit(rng));
    std::vector<bool> in_place;
    for (const int algo : {0, 4}) {
        SCOPED_TRACE("ALGO=" + std::to_string(algo));
        AlgoEnvGuard pin(algo);
        reset_grp_matmul_caches();
        std::vector<uint16_t> w = raw;
        std::vector<uint16_t> dst(static_cast<size_t>(M) * N / 2);
        mm::matmul_params p;
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::bf16;
        p.dtypes.dst = data_type_t::bf16;
        p.weight_cache_type = 2;
        mm::matmul_gated_act_params gated;
        gated.act = mm::matmul_gated_act_t::silu_and_mul;
        mm::matmul_batch_params_t batch;
        ASSERT_EQ(mm::matmul_direct('r', false, true, M, N, K, 1.0f, src.data(),
                          K, w.data(), K, nullptr, 0.0f, dst.data(), N / 2,
                          true, batch, p, &gated),
                status_t::success);
        in_place.push_back(w != raw);
        reset_grp_matmul_caches();
    }
    EXPECT_TRUE(in_place[0])
            << "the call packs in place without N-tile flat parallel";
    EXPECT_EQ(in_place[1], in_place[0]);
}

// Calls of one layer that give rows to other experts, named as zentorch
// names them (those with rows first, the rest as extras without their
// scales), keep one pool per projection and one call site.
TEST_F(NTileFlatParallelGrouped, RotatingActiveExpertsKeepOnePoolEach) {
    grouped_case c;
    c.wei = data_type_t::s8;
    c.fused = true;
    c.M = {2, 3, 4, 1};
    const grouped_inputs in = make_inputs(c);
    AlgoEnvGuard pin(4);
    reset_grp_matmul_caches();
    const std::vector<std::vector<int>> orders
            = {{0, 1, 2, 3}, {2, 3, 0, 1}, {1, 3, 0, 2}, {3, 0, 2, 1}};
    for (const std::vector<int> &order : orders) {
        ASSERT_EQ(run(c, in, order, /*active=*/2, /*bare_extras=*/true).status,
                status_t::success);
    }
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_warmed_pools(), 2u);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_call_sites(), 1u);
    reset_grp_matmul_caches();
}

// Per-tensor and F16 weight scales write what per-column F32 scales of the
// same values write, and one buffer given as per-tensor scales and then as
// per-column ones gives the per-column result.
TEST_F(NTileFlatParallelGrouped, PerTensorAndF16ScalesMatchPerColumnF32Scales) {
    AlgoEnvGuard pin(4);
    for (const data_type_t wei : {data_type_t::s8, data_type_t::s4}) {
        for (const bool fused : {false, true}) {
            SCOPED_TRACE(std::string(wei == data_type_t::s8 ? "da8w8" : "w4a8")
                    + (fused ? " fused" : ""));
            grouped_case c;
            c.wei = wei;
            c.fused = fused;
            const grouped_inputs in = make_inputs(c);
            const int n = static_cast<int>(c.M.size());
            // Per operation and projection: one value, exact in F16, as F32
            // per tensor and per column and as F16 per column; and a buffer
            // whose columns differ.
            std::vector<float> t13(n), t2(n);
            std::vector<std::vector<float>> col13(n), col2(n), vary13(n),
                    vary2(n);
            std::vector<std::vector<uint16_t>> h13(n), h2(n);
            for (int i = 0; i < n; ++i) {
                t13[i] = std::ldexp(1.0f + 0.25f * (i % 4), -7);
                t2[i] = std::ldexp(1.0f + 0.25f * ((i + 1) % 4), -7);
                col13[i].assign(c.N, t13[i]);
                col2[i].assign(c.N_down, t2[i]);
                h13[i].assign(c.N, float16_t::f32_to_f16_val(t13[i]));
                h2[i].assign(c.N_down, float16_t::f32_to_f16_val(t2[i]));
                vary13[i].resize(c.N);
                for (int j = 0; j < c.N; ++j)
                    vary13[i][j] = t13[i] * (1 + j % 3);
                vary2[i].resize(c.N_down);
                for (int j = 0; j < c.N_down; ++j)
                    vary2[i][j] = t2[i] * (1 + j % 3);
            }
            // 0: F32 per tensor, 1: F32 per column, 2: F16 per column, 3: F16
            // per tensor, 4: the varying buffer per tensor, 5: per column.
            const auto form = [&](const int f) {
                return [&, f](call_args &a) {
                    const auto set =
                            [f](mm::matmul_quantization_params_t::matmul_quant_t
                                            &q,
                                    float *t, float *col, uint16_t *h,
                                    float *vary, const int cols) {
                        const bool tensor = f == 0 || f == 3 || f == 4;
                        q.dt = f == 2 || f == 3 ? data_type_t::f16
                                                : data_type_t::f32;
                        q.buff = f == 0  ? static_cast<void *>(t)
                                : f == 1 ? static_cast<void *>(col)
                                : f >= 4 ? static_cast<void *>(vary)
                                         : static_cast<void *>(h);
                        q.dims = tensor ? std::vector<int64_t> {1}
                                        : std::vector<int64_t> {1, cols};
                    };
                    for (int i = 0; i < n; ++i) {
                        set(a.params[i].quant_params.wei_scale, &t13[i],
                                col13[i].data(), h13[i].data(),
                                vary13[i].data(), c.N);
                    }
                    for (int i = 0; fused && i < n; ++i) {
                        auto &d = a.fused.down_scale[i];
                        mm::matmul_quantization_params_t::matmul_quant_t q;
                        set(q, &t2[i], col2[i].data(), h2[i].data(),
                                vary2[i].data(), c.N_down);
                        d.buff = q.buff;
                        d.dt = q.dt;
                        d.dims = q.dims;
                    }
                };
            };
            reset_grp_matmul_caches();
            const grouped_result per_column
                    = run(c, in, {}, -1, false, form(1));
            ASSERT_EQ(per_column.status, status_t::success);
            for (const int f : {0, 2, 3}) {
                SCOPED_TRACE("form " + std::to_string(f));
                reset_grp_matmul_caches();
                const grouped_result r = run(c, in, {}, -1, false, form(f));
                ASSERT_EQ(r.status, status_t::success);
                EXPECT_EQ(r.values, per_column.values);
            }
            reset_grp_matmul_caches();
            const grouped_result fresh = run(c, in, {}, -1, false, form(5));
            ASSERT_EQ(fresh.status, status_t::success);
            reset_grp_matmul_caches();
            ASSERT_EQ(run(c, in, {}, -1, false, form(4)).status,
                    status_t::success);
            const grouped_result after = run(c, in, {}, -1, false, form(5));
            ASSERT_EQ(after.status, status_t::success);
            EXPECT_EQ(after.values, fresh.values)
                    << "the per-column call read the per-tensor call's pack";
        }
    }
    reset_grp_matmul_caches();
}

// A flush during a call, right after the call read the flush count on
// entering the API or between finding its weights and publishing their
// packs, leaves the call packs of its own: it succeeds, writing what it
// writes alone, and leaves no pack or call site behind, so the host may
// reload the weights in place once both return.
TEST_F(NTileFlatParallelGrouped, AFlushDuringACallLeavesNothingBehind) {
    for (const bool after_entry : {false, true}) {
        SCOPED_TRACE(after_entry ? "flush after the call read the flush count"
                                 : "flush before the call publishes");
        grouped_case c;
        c.wei = data_type_t::s8;
        c.fused = true;
        grouped_inputs in = make_inputs(c);
        std::vector<std::vector<float>> up, down;
        for (const auto &w : in.w13)
            up.push_back(w.scale);
        for (const auto &w : in.w2)
            down.push_back(w.scale);
        const auto moved = [&](call_args &a) {
            for (size_t i = 0; i < a.params.size(); ++i)
                a.params[i].quant_params.wei_scale.buff = up[i].data();
            for (size_t i = 0; i < a.fused.down_scale.size(); ++i)
                a.fused.down_scale[i].buff = down[i].data();
        };
        AlgoEnvGuard pin(4);
        reset_grp_matmul_caches();
        const grouped_result first = run(c, in);
        ASSERT_EQ(first.status, status_t::success);
        auto &flush = after_entry
                ? mm::test_api::s_ntile_flat_parallel_flush_after_entry
                : mm::test_api::s_ntile_flat_parallel_flush_before_publish;
        {
            hook_pin armed(flush, true);
            const grouped_result r = run(c, in, {}, -1, false, moved);
            EXPECT_FALSE(flush.load()) << "the flush did not run";
            EXPECT_EQ(r.status, status_t::success);
            EXPECT_EQ(r.values, first.values);
        }
        EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 0u);
        EXPECT_EQ(mm::test_api::ntile_flat_parallel_call_sites(), 0u);
        EXPECT_EQ(mm::test_api::s_ntile_flat_parallel_live_packs.load(), 0);
        const grouped_inputs reload = make_inputs(c, 11);
        for (size_t e = 0; e < in.w13.size(); ++e) {
            std::copy(reload.w13[e].data.begin(), reload.w13[e].data.end(),
                    in.w13[e].data.begin());
            std::copy(reload.w2[e].data.begin(), reload.w2[e].data.end(),
                    in.w2[e].data.begin());
            std::copy(reload.w13[e].scale.begin(), reload.w13[e].scale.end(),
                    up[e].begin());
            std::copy(reload.w2[e].scale.begin(), reload.w2[e].scale.end(),
                    down[e].begin());
        }
        const grouped_result reloaded = run(c, in, {}, -1, false, moved);
        ASSERT_EQ(reloaded.status, status_t::success);
        EXPECT_NE(reloaded.values, first.values);
        reset_grp_matmul_caches();
        const grouped_result fresh = run(c, in, {}, -1, false, moved);
        ASSERT_EQ(fresh.status, status_t::success);
        EXPECT_EQ(reloaded.values, fresh.values)
                << "the call ran on packs of the weights before the reload";
    }
    reset_grp_matmul_caches();
}

/// What `call_during_flush` runs.
std::function<void()> g_during_flush;
void call_during_flush() {
    g_during_flush();
}

// A call made while a flush runs, after the flush has dropped the store,
// packs its weights for itself: it writes what it writes alone and leaves no
// pack or call site behind.
TEST_F(NTileFlatParallelGrouped, ACallDuringAFlushLeavesNothingBehind) {
    grouped_case c;
    c.wei = data_type_t::s8;
    c.fused = true;
    const grouped_inputs in = make_inputs(c);
    AlgoEnvGuard pin(4);
    reset_grp_matmul_caches();
    grouped_result during;
    g_during_flush = [&] { during = run(c, in); };
    {
        hook_pin armed(mm::test_api::s_ntile_flat_parallel_during_flush,
                &call_during_flush);
        reset_grp_matmul_caches();
        EXPECT_EQ(mm::test_api::s_ntile_flat_parallel_during_flush.load(),
                nullptr)
                << "the flush did not make the call";
    }
    g_during_flush = nullptr;
    ASSERT_EQ(during.status, status_t::success);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 0u);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_call_sites(), 0u);
    EXPECT_EQ(mm::test_api::s_ntile_flat_parallel_live_packs.load(), 0);
    const grouped_result after = run(c, in);
    ASSERT_EQ(after.status, status_t::success);
    EXPECT_EQ(during.values, after.values);
    reset_grp_matmul_caches();
}

} // namespace
