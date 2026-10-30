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

/// HSTU attention: the causal row band against the mask predicate, both
/// kernels against an FP64 transcription of `pytorch_hstu_mha` /
/// `pytorch_cached_hstu_mha`, the bitwise determinism of the tiled kernel and
/// its fast silu (ZENDNNL_HSTU_ATTENTION_FAST_SILU).

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/hstu_attention/common/hstu_attention_pad.hpp"
#include "lowoha_operators/hstu_attention/common/hstu_attention_plan.hpp"
#include "lowoha_operators/hstu_attention/common/hstu_attention_tiled_driver.hpp"
#include "lowoha_operators/hstu_attention/hstu_attention_mask.hpp"
#include "lowoha_operators/hstu_attention/lowoha_hstu_attention.hpp"
#include "lowoha_operators/hstu_attention/tiled/hstu_attention_tiled.hpp"

#include <omp.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <vector>
#include <initializer_list>

namespace {

using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;
using zendnnl::common::status_t;
namespace ha = zendnnl::lowoha::hstu_attention;
using kernel_t = ha::hstu_attention_kernel_t;

bool has_bf16() {
    const auto &pi = zendnnl::common::zendnnl_platform_info();
    return pi.get_avx512_bf16_status() && pi.get_avx512_bw_vl_status();
}

/// `_get_valid_attn_mask` of generative-recommenders for one sequence;
/// `targets` < 0 means no num_targets tensor.
bool oracle_valid(int64_t n, int64_t targets, int64_t ctx, int64_t window,
        int64_t min_full, bool causal, int64_t i, int64_t j) {
    int64_t max_ids = n;
    if (ctx > 0) max_ids = max_ids - ctx + 1;
    if (targets >= 0) max_ids -= targets;
    auto id = [&](int64_t x) {
        if (ctx > 0) x = std::max<int64_t>(0, x - ctx + 1);
        if (targets >= 0) x = std::min(x, max_ids);
        return x;
    };
    int64_t d = id(i) - id(j);
    if (!causal) d = std::abs(d);
    bool ok = i == j || d > 0;
    if (window > 0) {
        if (min_full > 0)
            ok = ok && (d <= window || id(i) >= max_ids - min_full);
        else
            ok = ok && d <= window;
    }
    if (ctx > 0) ok = ok || (id(i) == 0 && id(j) < max_ids);
    return ok;
}

using i64_list = std::initializer_list<int64_t>;

TEST(HstuAttentionMask, CausalBandMatchesPredicate) {
    for (int64_t n = 1; n <= 40; ++n)
        for (int64_t t : i64_list {-1, 0, 1, 3, n - 1, n})
            for (int64_t c : i64_list {0, 1, 3})
                for (int64_t w : i64_list {0, 1, 4, 100})
                    for (int64_t m : i64_list {0, 2}) {
                        if (t > n || t < -1) continue;
                        ha::hstu_attention_params p;
                        p.causal = true;
                        p.contextual_seq_len = c;
                        p.max_attn_len = w;
                        p.min_full_attn_seq_len = m;
                        const ha::seq_mask_t mask(p, n, t);
                        for (int64_t i = 0; i < n; ++i) {
                            const ha::row_band_t band = mask.band(i);
                            for (int64_t j = 0; j < n; ++j) {
                                const bool want = oracle_valid(
                                        n, t, c, w, m, true, i, j);
                                ASSERT_EQ(mask.valid(i, j), want)
                                        << "n=" << n << " t=" << t << " c=" << c
                                        << " w=" << w << " m=" << m
                                        << " i=" << i << " j=" << j;
                                const bool in_band
                                        = (j >= band.lo && j < band.hi)
                                        || (band.extra_diag && j == i);
                                ASSERT_EQ(in_band, want)
                                        << "n=" << n << " t=" << t << " c=" << c
                                        << " w=" << w << " m=" << m
                                        << " i=" << i << " j=" << j;
                            }
                        }
                    }
}

TEST(HstuAttentionMask, NonCausalPredicate) {
    for (int64_t n = 1; n <= 24; ++n)
        for (int64_t t : i64_list {-1, 0, 2, n})
            for (int64_t c : i64_list {0, 2, n + 1})
                for (int64_t w : i64_list {0, 1, 5})
                    for (int64_t m : i64_list {0, 3}) {
                        ha::hstu_attention_params p;
                        p.causal = false;
                        p.contextual_seq_len = c;
                        p.max_attn_len = w;
                        p.min_full_attn_seq_len = m;
                        const ha::seq_mask_t mask(p, n, t);
                        for (int64_t i = 0; i < n; ++i)
                            for (int64_t j = 0; j < n; ++j)
                                ASSERT_EQ(mask.valid(i, j),
                                        oracle_valid(
                                                n, t, c, w, m, false, i, j))
                                        << "n=" << n << " t=" << t << " c=" << c
                                        << " w=" << w << " m=" << m
                                        << " i=" << i << " j=" << j;
                    }
}

/// Kernel selection and validation can be forced through the environment,
/// which would change what these tests exercise.
bool env_overrides_kernel() {
    const char *k = std::getenv("ZENDNNL_HSTU_ATTENTION_KERNEL");
    return k && *k;
}

struct hstu_case {
    std::string name;
    std::vector<int64_t> lens;
    int64_t heads = 2, qk_dim = 32, v_dim = 48;
    std::vector<int64_t> targets; ///< empty: no num_targets
    int64_t window = 0, ctx = 0, min_full = 0;
    bool causal = true;
    int64_t delta = 0;
    data_type_t in_dt = data_type_t::bf16, out_dt = data_type_t::bf16;
    bool fused_uvqk = false; ///< q/k/v as head slices of one [L, W] buffer
    bool per_token_scale = false;
    bool has_scale = false; ///< scalar attn_scale = scale_value
    float scale_value = 0.f;
    int64_t base = 0; ///< seq_offsets[0]; rows before it must stay untouched
    float input_scale = 1.f; ///< inputs are bf16(2 * input_scale * U(-1, 1))
    int64_t head_gap = 0; ///< unused elements after each head of Q/K/V, out
};

std::ostream &operator<<(std::ostream &os, const hstu_case &c) {
    return os << c.name;
}

std::vector<hstu_case> make_cases() {
    std::vector<hstu_case> cs;
    auto add = [&](hstu_case c) { cs.push_back(std::move(c)); };
    add({"causal", {1, 7, 64, 130, 257}});
    add({"targets", {50, 200, 129, 3}, 2, 32, 48, {5, 0, 129, 3}});
    {
        hstu_case c {"window", {300, 129, 64}, 2, 32, 48, {7, 0, 9}};
        c.window = 17;
        add(c);
        c.name = "window_min_full";
        c.min_full = 9;
        add(c);
    }
    {
        hstu_case c {"contextual", {90, 260, 8}, 2, 32, 48, {6, 31, 0}};
        c.ctx = 3;
        add(c);
        c.name = "contextual_window";
        c.window = 20;
        add(c);
    }
    {
        hstu_case c {"noncausal", {70, 131}, 2, 32, 48, {4, 10}};
        c.causal = false;
        add(c);
    }
    {
        hstu_case c {"delta", {40, 300, 129}, 2, 32, 48, {16, 4, 0}};
        c.delta = 16;
        add(c);
        c.name = "delta_window";
        c.window = 25;
        add(c);
    }
    {
        hstu_case c {"f32", {33, 150}, 2, 16, 24, {3, 20}};
        c.in_dt = data_type_t::f32;
        c.out_dt = data_type_t::f32;
        add(c);
    }
    {
        hstu_case c {"bf16_in_f32_out", {33, 150}, 2, 16, 24, {3, 20}};
        c.out_dt = data_type_t::f32;
        add(c);
    }
    add({"odd_dims", {65, 300, 31}, 3, 33, 41, {7, 0, 31}});
    {
        hstu_case c {"head_gap", {65, 300, 0, 31}, 3, 32, 48, {7, 0, 0, 31}};
        c.head_gap = 8;
        add(c);
        c.name = "head_gap_fused_uvqk";
        c.fused_uvqk = true;
        add(c);
    }
    {
        hstu_case c {"per_token_scale", {77, 140}, 2, 32, 32, {5, 0}};
        c.per_token_scale = true;
        add(c);
    }
    {
        // Several query tiles per sequence in delta mode (delta > 32, not a
        // multiple of 32).
        hstu_case c {
                "delta_multi_tile", {300, 1100, 900}, 2, 32, 48, {16, 300, 0}};
        c.delta = 200;
        add(c);
        hstu_case w {"delta_multi_tile_window", {700, 1100, 650}, 2, 32, 48,
                {50, 600, 10}};
        w.delta = 600;
        w.window = 300;
        add(w);
    }
    {
        hstu_case c {"window_long", {1100, 700}, 2, 32, 48, {64, 10}};
        c.window = 300;
        add(c);
    }
    for (float sv : {0.37f, 0.f, -0.5f}) {
        hstu_case c {sv > 0 ? "scale_positive"
                            : (sv == 0 ? "scale_zero" : "scale_negative"),
                {77, 140}, 2, 32, 32, {5, 0}};
        c.has_scale = true;
        c.scale_value = sv;
        add(c);
    }
    {
        hstu_case c {"empty_sequences_offset_base", {0, 300, 0, 5, 257}, 2, 32,
                48, {0, 20, 0, 5, 1}};
        c.base = 37;
        add(c);
        c.name = "empty_sequences_offset_base_window";
        c.window = 40;
        add(c);
    }
    {
        hstu_case c {"dlrm_shape_fused_uvqk", {700, 1100, 260}, 4, 128, 128,
                {64, 32, 0}};
        c.fused_uvqk = true;
        add(c);
    }
    return cs;
}

/// Tensors of one case; `values` holds the inputs as the kernel sees them
/// (rounded to bf16 for bf16 inputs).
struct hstu_problem {
    explicit hstu_problem(const hstu_case &c) : cs(c) {
        const int64_t B = c.lens.size(), H = c.heads;
        offsets.assign(B + 1, c.base);
        for (int64_t b = 0; b < B; ++b)
            offsets[b + 1] = offsets[b] + c.lens[b];
        const int64_t L = offsets[B];
        max_len = *std::max_element(c.lens.begin(), c.lens.end()) + 5;
        q_rows = c.delta > 0 ? B * c.delta : L;
        first_row = c.delta > 0 ? 0 : c.base;

        // Q rows live in their own buffer in delta mode; K and V (and Q in
        // full mode) are slices of one buffer when fused_uvqk.
        const int64_t qk = H * (c.qk_dim + c.head_gap),
                      hv = H * (c.v_dim + c.head_gap);
        if (c.fused_uvqk) {
            width = 2 * hv + 2 * qk + 8; // u, v, q, k and padding
            v_off = hv;
            q_off = 2 * hv;
            k_off = 2 * hv + qk;
        } else {
            width = qk + qk + hv;
            q_off = 0;
            k_off = qk;
            v_off = 2 * qk;
        }
        std::mt19937 gen(1234);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        values.resize(L * width);
        for (float &x : values)
            x = round(2.f * c.input_scale * dist(gen));
        if (c.delta > 0) {
            dq.resize(q_rows * qk);
            for (float &x : dq)
                x = round(2.f * c.input_scale * dist(gen));
        }
        if (c.per_token_scale) {
            scale.resize(L);
            for (float &x : scale)
                x = 0.01f + 0.02f * (dist(gen) + 1.f);
        }
        alpha = 1.f / std::sqrt(static_cast<float>(c.qk_dim));
        o_stride_t = H * (c.v_dim + c.head_gap) + 16; // gaps are canaries
    }

    float round(float x) const {
        return cs.in_dt == data_type_t::bf16 ? static_cast<float>(bfloat16_t(x))
                                             : x;
    }

    float q_at(int64_t row, int64_t h, int64_t d) const {
        if (cs.delta > 0) return dq[(row * cs.heads + h) * cs.qk_dim + d];
        return values[row * width + q_off + h * (cs.qk_dim + cs.head_gap) + d];
    }
    float k_at(int64_t row, int64_t h, int64_t d) const {
        return values[row * width + k_off + h * (cs.qk_dim + cs.head_gap) + d];
    }
    float v_at(int64_t row, int64_t h, int64_t d) const {
        return values[row * width + v_off + h * (cs.v_dim + cs.head_gap) + d];
    }

    /// FP64 output [q_rows, H, v_dim].
    std::vector<double> oracle() const {
        const int64_t B = cs.lens.size(), H = cs.heads;
        std::vector<double> out(q_rows * H * cs.v_dim, 0.0);
        for (int64_t b = 0; b < B; ++b) {
            const int64_t n = cs.lens[b], base = offsets[b];
            const int64_t t = cs.targets.empty() ? -1 : cs.targets[b];
            const int64_t nq = cs.delta > 0 ? cs.delta : n;
            const int64_t pos0 = cs.delta > 0 ? n - cs.delta : 0;
            const int64_t qrow0 = cs.delta > 0 ? b * cs.delta : base;
            for (int64_t h = 0; h < H; ++h)
                for (int64_t r = 0; r < nq; ++r) {
                    const int64_t i = pos0 + r;
                    const double sc = cs.per_token_scale ? scale[base + i]
                            : cs.has_scale               ? cs.scale_value
                                           : 1.0 / static_cast<double>(max_len);
                    double *o = &out[((qrow0 + r) * H + h) * cs.v_dim];
                    for (int64_t j = 0; j < n; ++j) {
                        if (!oracle_valid(n, t, cs.ctx, cs.window, cs.min_full,
                                    cs.causal, i, j))
                            continue;
                        double s = 0.0;
                        for (int64_t d = 0; d < cs.qk_dim; ++d)
                            s += double(q_at(qrow0 + r, h, d))
                                    * k_at(base + j, h, d);
                        const double x = alpha * s;
                        const double w = x / (1.0 + std::exp(-x)) * sc;
                        for (int64_t d = 0; d < cs.v_dim; ++d)
                            o[d] += w * v_at(base + j, h, d);
                    }
                }
        }
        return out;
    }

    /// Runs the op; returns the output as float [q_rows, H, v_dim] (rows
    /// before first_row as 0) and whether every canary (rows before
    /// first_row, row padding) is untouched. `fast_silu` >= 0 calls the tiled
    /// kernel directly with that silu, bypassing the environment. `varlen`
    /// is params.is_varlen.
    status_t run(kernel_t kernel, std::vector<float> &result,
            int32_t num_threads = 0, bool *canary_ok = nullptr,
            int fast_silu = -1, bool varlen = true) const {
        const int64_t H = cs.heads;
        const bool bf = cs.in_dt == data_type_t::bf16;
        std::vector<bfloat16_t> kv_bf, dq_bf;
        if (bf) {
            kv_bf.assign(values.begin(), values.end());
            dq_bf.assign(dq.begin(), dq.end());
        }
        const size_t esz = bf ? sizeof(bfloat16_t) : sizeof(float);
        const char *kv = bf ? reinterpret_cast<const char *>(kv_bf.data())
                            : reinterpret_cast<const char *>(values.data());
        const char *qbuf = cs.delta > 0
                ? (bf ? reinterpret_cast<const char *>(dq_bf.data())
                      : reinterpret_cast<const char *>(dq.data()))
                : kv + q_off * esz;

        ha::hstu_attention_params p;
        p.num_seqs = cs.lens.size();
        p.num_heads = H;
        p.qk_dim = cs.qk_dim;
        p.v_dim = cs.v_dim;
        p.max_seq_len = max_len;
        p.seq_offsets = offsets.data();
        p.delta_size = cs.delta;
        p.causal = cs.causal;
        p.num_targets = cs.targets.empty() ? nullptr : cs.targets.data();
        p.max_attn_len = cs.window;
        p.contextual_seq_len = cs.ctx;
        p.min_full_attn_seq_len = cs.min_full;
        p.alpha = alpha;
        p.attn_scale_per_token = cs.per_token_scale ? scale.data() : nullptr;
        p.has_attn_scale = cs.has_scale;
        p.attn_scale = cs.scale_value;
        p.num_threads = num_threads;
        p.q_stride_t = cs.delta > 0 ? H * cs.qk_dim : width;
        p.q_stride_h = cs.delta > 0 ? cs.qk_dim : cs.qk_dim + cs.head_gap;
        p.k_stride_t = width;
        p.k_stride_h = cs.qk_dim + cs.head_gap;
        p.v_stride_t = width;
        p.v_stride_h = cs.v_dim + cs.head_gap;
        p.o_stride_t = o_stride_t;
        p.o_stride_h = cs.v_dim + cs.head_gap;
        p.qkv_dt = cs.in_dt;
        p.out_dt = cs.out_dt;
        p.kernel = kernel;
        p.is_varlen = varlen;

        const size_t out_elems = q_rows * o_stride_t;
        std::vector<float> out_f(out_elems, -7.f);
        std::vector<bfloat16_t> out_b(out_elems, bfloat16_t(-7.f));
        void *out = cs.out_dt == data_type_t::f32
                ? static_cast<void *>(out_f.data())
                : static_cast<void *>(out_b.data());
        const status_t st = fast_silu < 0
                ? ha::hstu_attention_direct(
                          qbuf, kv + k_off * esz, kv + v_off * esz, out, p)
                : ha::hstu_attention_tiled(qbuf, kv + k_off * esz,
                          kv + v_off * esz, out, p, fast_silu == 1);
        auto at = [&](size_t i) {
            return cs.out_dt == data_type_t::f32 ? out_f[i]
                                                 : static_cast<float>(out_b[i]);
        };
        bool untouched = true;
        result.assign(q_rows * H * cs.v_dim, 0.f);
        const int64_t hs = cs.v_dim + cs.head_gap;
        for (int64_t r = 0; r < q_rows; ++r) {
            for (int64_t e = 0; e < o_stride_t; ++e) {
                const float x = at(r * o_stride_t + e);
                const int64_t h = e / hs, d = e % hs;
                if (r < first_row || h >= H || d >= cs.v_dim)
                    untouched = untouched && x == -7.f;
                else
                    result[(r * H + h) * cs.v_dim + d] = x;
            }
        }
        if (canary_ok) *canary_ok = untouched;
        return st;
    }

    const hstu_case &cs;
    std::vector<int64_t> offsets;
    int64_t max_len = 0, q_rows = 0, first_row = 0, width = 0;
    int64_t q_off = 0, k_off = 0, v_off = 0, o_stride_t = 0;
    float alpha = 1.f;
    std::vector<float> values, dq, scale;
};

/// max |got - want| / max |want|; +inf when any value is not finite.
double rel_error(
        const std::vector<float> &got, const std::vector<double> &want) {
    double err = 0.0, mag = 1e-30;
    for (size_t i = 0; i < want.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i]))
            return std::numeric_limits<double>::infinity();
        err = std::max(err, std::abs(got[i] - want[i]));
        mag = std::max(mag, std::abs(want[i]));
    }
    return err / mag;
}

class HstuAttentionKernels : public ::testing::TestWithParam<hstu_case> {};

TEST_P(HstuAttentionKernels, MatchOracle) {
    if (env_overrides_kernel())
        GTEST_SKIP() << "ZENDNNL_HSTU_ATTENTION_KERNEL overrides the kernel";
    const hstu_case &c = GetParam();
    const hstu_problem prob(c);
    const std::vector<double> want = prob.oracle();
    const double tol
            = c.out_dt == data_type_t::f32 && c.in_dt == data_type_t::f32
            ? 1e-5
            : 1.5e-2;

    std::vector<float> got;
    bool canary = false;
    ASSERT_EQ(
            prob.run(kernel_t::reference, got, 0, &canary), status_t::success);
    EXPECT_LT(rel_error(got, want), tol) << "reference";
    EXPECT_TRUE(canary) << "reference wrote outside its rows";

    const bool tiled = c.causal && c.in_dt == data_type_t::bf16 && has_bf16();
    // The tiled kernel sizes its query tiles from the thread count.
    for (int32_t threads : {0, 1, 5, 64}) {
        ASSERT_EQ(prob.run(kernel_t::none, got, threads, &canary),
                status_t::success);
        EXPECT_LT(rel_error(got, want), tol) << "auto, threads=" << threads;
        EXPECT_TRUE(canary) << "auto wrote outside its rows";

        const status_t st = prob.run(kernel_t::tiled, got, threads, &canary);
        if (!tiled) {
            EXPECT_EQ(st, status_t::unimplemented);
            continue;
        }
        ASSERT_EQ(st, status_t::success);
        EXPECT_LT(rel_error(got, want), tol) << "tiled, threads=" << threads;
        EXPECT_TRUE(canary) << "tiled wrote outside its rows";
    }
}

INSTANTIATE_TEST_SUITE_P(Cases, HstuAttentionKernels,
        ::testing::ValuesIn(make_cases()),
        [](const ::testing::TestParamInfo<hstu_case> &info) {
            return info.param.name;
        });

hstu_case find_case(const std::string &name) {
    for (const hstu_case &c : make_cases())
        if (c.name == name) return c;
    ADD_FAILURE() << "no case " << name;
    return {};
}

bool bitwise_equal(const std::vector<float> &a, const std::vector<float> &b) {
    return a.size() == b.size()
            && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

/// Rows are independent and P columns outside a row's band are exact zeros,
/// so the tiled output does not depend on the tile size or thread count.
TEST(HstuAttentionTiled, DeterministicAcrossThreads) {
    if (env_overrides_kernel() || !has_bf16()) GTEST_SKIP();
    for (const hstu_case &c : make_cases()) {
        if (!c.causal || c.in_dt != data_type_t::bf16) continue;
        const hstu_problem prob(c);
        std::vector<float> want, got;
        ASSERT_EQ(prob.run(kernel_t::tiled, want, 1), status_t::success);
        for (int32_t threads : {5, 64}) {
            ASSERT_EQ(
                    prob.run(kernel_t::tiled, got, threads), status_t::success);
            EXPECT_TRUE(bitwise_equal(got, want))
                    << c.name << ", threads=" << threads;
        }
    }
}

/// Scratch is reused across calls (and output types); results must not
/// depend on what an earlier call left in it.
TEST(HstuAttentionTiled, RepeatedCallsReuseScratch) {
    if (env_overrides_kernel() || !has_bf16()) GTEST_SKIP();
    const hstu_case big = find_case("dlrm_shape_fused_uvqk");
    hstu_case delta = find_case("delta_window");
    hstu_case delta_f32 = delta;
    delta_f32.out_dt = data_type_t::f32;
    const hstu_problem big_prob(big), delta_prob(delta),
            delta_f32_prob(delta_f32);
    std::vector<float> big0, delta0, delta_f32_0, got;
    for (int32_t threads : {0, 3}) {
        ASSERT_EQ(big_prob.run(kernel_t::tiled, big0, threads),
                status_t::success);
        ASSERT_EQ(delta_prob.run(kernel_t::tiled, delta0, threads),
                status_t::success);
        ASSERT_EQ(delta_f32_prob.run(kernel_t::tiled, delta_f32_0, threads),
                status_t::success);
        for (int rep = 0; rep < 2; ++rep) {
            ASSERT_EQ(big_prob.run(kernel_t::tiled, got, threads),
                    status_t::success);
            EXPECT_TRUE(bitwise_equal(got, big0)) << "big, rep " << rep;
            ASSERT_EQ(delta_f32_prob.run(kernel_t::tiled, got, threads),
                    status_t::success);
            EXPECT_TRUE(bitwise_equal(got, delta_f32_0))
                    << "delta f32, rep " << rep;
            ASSERT_EQ(delta_prob.run(kernel_t::tiled, got, threads),
                    status_t::success);
            EXPECT_TRUE(bitwise_equal(got, delta0)) << "delta, rep " << rep;
        }
    }
}

TEST(HstuAttentionFastSilu, ParsesEnv) {
    EXPECT_EQ(ha::parse_fast_silu_env(nullptr), -1);
    EXPECT_EQ(ha::parse_fast_silu_env(""), -1);
    EXPECT_EQ(ha::parse_fast_silu_env("2"), -1);
    EXPECT_EQ(ha::parse_fast_silu_env("fast"), -1);
    for (const char *on : {"1", "true", "TRUE", "on", "yes"})
        EXPECT_EQ(ha::parse_fast_silu_env(on), 1) << on;
    for (const char *off : {"0", "false", "Off", "no"})
        EXPECT_EQ(ha::parse_fast_silu_env(off), 0) << off;
}

TEST(HstuAttentionVarlen, ParsesEnv) {
    EXPECT_EQ(ha::parse_varlen_env(nullptr), -1);
    EXPECT_EQ(ha::parse_varlen_env(""), -1);
    EXPECT_EQ(ha::parse_varlen_env("pad"), -1);
    for (const char *on : {"1", "true", "ON", "yes"})
        EXPECT_EQ(ha::parse_varlen_env(on), 1) << on;
    for (const char *off : {"0", "false", "Off", "no"})
        EXPECT_EQ(ha::parse_varlen_env(off), 0) << off;
}

/// Padding every sequence to max_seq_len (is_varlen = false) gives the
/// variable-length output bit for bit, on every case (delta mode stays
/// variable-length) and kernel, and leaves the canaries untouched.
TEST(HstuAttentionVarlen, PaddedMatchesVarlen) {
    if (env_overrides_kernel()) GTEST_SKIP();
    for (const hstu_case &c : make_cases()) {
        const hstu_problem prob(c);
        for (kernel_t kernel : {kernel_t::none, kernel_t::reference})
            for (int32_t threads : {1, 5}) {
                std::vector<float> var, pad;
                bool var_ok = false, pad_ok = false;
                ASSERT_EQ(prob.run(kernel, var, threads, &var_ok),
                        status_t::success)
                        << c.name;
                ASSERT_EQ(prob.run(kernel, pad, threads, &pad_ok, -1, false),
                        status_t::success)
                        << c.name;
                EXPECT_TRUE(pad_ok) << c.name;
                EXPECT_TRUE(bitwise_equal(var, pad))
                        << c.name << ", kernel " << ha::kernel_to_string(kernel)
                        << ", threads " << threads;
            }
    }
}

/// The tiled cases, plus scores large enough (|alpha q.k| beyond 100) that
/// exp(-x) saturates to 0 and inf.
std::vector<hstu_case> fast_silu_cases() {
    std::vector<hstu_case> cs;
    for (const hstu_case &c : make_cases()) {
        if (c.causal && c.in_dt == data_type_t::bf16) cs.push_back(c);
    }
    hstu_case big {"large_scores", {300, 129}, 2, 32, 48, {7, 0}};
    big.input_scale = 4.f;
    cs.push_back(big);
    big.name = "large_scores_f32_out";
    big.out_dt = data_type_t::f32;
    cs.push_back(big);
    return cs;
}

/// The fast silu adds at most 1.2e-4 relative error to P before its BF16
/// rounding (2^-9), so the outputs keep the default path's error.
TEST(HstuAttentionFastSilu, MatchesOracle) {
    if (!has_bf16()) GTEST_SKIP();
    for (const hstu_case &c : fast_silu_cases()) {
        const hstu_problem prob(c);
        const std::vector<double> want = prob.oracle();
        std::vector<float> base, got;
        ASSERT_EQ(prob.run(kernel_t::tiled, base, 0, nullptr, 0),
                status_t::success);
        const double base_err = rel_error(base, want);
        EXPECT_LT(base_err, 1.5e-2) << c.name << ", default silu";
        for (int32_t threads : {0, 1, 5, 64}) {
            bool canary = false;
            ASSERT_EQ(prob.run(kernel_t::tiled, got, threads, &canary, 1),
                    status_t::success);
            const double err = rel_error(got, want);
            EXPECT_LT(err, 1.5e-2) << c.name << ", threads=" << threads;
            EXPECT_LT(err, 1.1 * base_err + 5e-4)
                    << c.name << ", threads=" << threads
                    << ", default silu error " << base_err;
            EXPECT_TRUE(canary) << c.name << " wrote outside its rows";
        }
    }
}

/// The default path is the explicit `fast_silu = false` one, bit for bit
/// (the golden dumps of the harness pin it across commits); the fast path is
/// deterministic across thread counts and differs from it.
TEST(HstuAttentionFastSilu, DefaultPathUnchanged) {
    if (env_overrides_kernel() || !has_bf16()) GTEST_SKIP();
    const char *env = std::getenv("ZENDNNL_HSTU_ATTENTION_FAST_SILU");
    const bool env_fast = ha::parse_fast_silu_env(env) == 1;
    for (const hstu_case &c : fast_silu_cases()) {
        const hstu_problem prob(c);
        std::vector<float> slow, fast, got;
        ASSERT_EQ(prob.run(kernel_t::tiled, slow, 1, nullptr, 0),
                status_t::success);
        ASSERT_EQ(prob.run(kernel_t::tiled, fast, 1, nullptr, 1),
                status_t::success);
        for (int32_t threads : {0, 5, 64}) {
            ASSERT_EQ(
                    prob.run(kernel_t::none, got, threads), status_t::success);
            EXPECT_TRUE(bitwise_equal(got, env_fast ? fast : slow))
                    << c.name << ", threads=" << threads;
            ASSERT_EQ(prob.run(kernel_t::tiled, got, threads, nullptr, 1),
                    status_t::success);
            EXPECT_TRUE(bitwise_equal(got, fast))
                    << c.name << ", fast, threads=" << threads;
        }
        if (c.name == "dlrm_shape_fused_uvqk" || c.name == "large_scores") {
            EXPECT_FALSE(bitwise_equal(fast, slow)) << c.name;
        }
    }
}

/// The fields of a case that the work decomposition reads; `c` and `offsets`
/// must outlive the result.
ha::hstu_attention_params plan_params(
        const hstu_case &c, std::vector<int64_t> &offsets) {
    offsets.assign(c.lens.size() + 1, c.base);
    for (size_t b = 0; b < c.lens.size(); ++b)
        offsets[b + 1] = offsets[b] + c.lens[b];
    ha::hstu_attention_params p;
    p.num_seqs = c.lens.size();
    p.num_heads = c.heads;
    p.seq_offsets = offsets.data();
    p.delta_size = c.delta;
    p.causal = c.causal;
    p.num_targets = c.targets.empty() ? nullptr : c.targets.data();
    p.max_attn_len = c.window;
    p.contextual_seq_len = c.ctx;
    p.min_full_attn_seq_len = c.min_full;
    p.v_dim = c.v_dim;
    p.max_seq_len = *std::max_element(c.lens.begin(), c.lens.end()) + 5;
    return p;
}

TEST(HstuAttentionPlan, WorkItemsInSdpaOrder) {
    for (const hstu_case &c : make_cases()) {
        if (!c.causal) continue;
        std::vector<int64_t> offsets;
        const ha::hstu_attention_params p = plan_params(c, offsets);
        for (int64_t bm : {32, 64, 128}) {
            const ha::work_plan_t plan(p, bm);
            int64_t z = 0;
            for (int64_t b = 0; b < p.num_seqs; ++b) {
                const ha::seq_rows_t s = ha::seq_rows(p, b);
                for (int64_t h = 0; h < p.num_heads; ++h)
                    for (int64_t row0 = 0; row0 < s.num_q; row0 += bm, ++z) {
                        ASSERT_LT(z, plan.size()) << c.name;
                        const ha::work_item_t w = plan.item(z);
                        ASSERT_EQ(w.b, b) << c.name << " z=" << z;
                        ASSERT_EQ(w.h, h) << c.name << " z=" << z;
                        ASSERT_EQ(w.row0, row0) << c.name << " z=" << z;
                        ASSERT_EQ(w.rows, std::min(bm, s.num_q - row0))
                                << c.name << " z=" << z;
                        ASSERT_EQ(w.s.q_row, s.q_row) << c.name;
                        ASSERT_EQ(w.s.num_q, s.num_q) << c.name;
                    }
            }
            EXPECT_EQ(z, plan.size()) << c.name << " bm=" << bm;
        }
    }
}

TEST(HstuAttentionPlan, KeyBlocksCoverKeyRanges) {
    for (const hstu_case &c : make_cases()) {
        if (!c.causal) continue;
        std::vector<int64_t> offsets;
        const ha::hstu_attention_params p = plan_params(c, offsets);
        const std::vector<ha::key_range_t> need = ha::key_ranges(p);
        ASSERT_EQ(int64_t(need.size()), p.num_seqs);
        // Every valid key except a diagonal outside the row's band.
        for (int64_t b = 0; b < p.num_seqs; ++b) {
            const ha::seq_rows_t s = ha::seq_rows(p, b);
            const ha::seq_mask_t mask(p, s.len, s.targets);
            int64_t lo = s.len, hi = 0;
            for (int64_t i = s.q_pos; i < s.q_pos + s.num_q; ++i) {
                const ha::row_band_t band = mask.band(i);
                for (int64_t j = 0; j < s.len; ++j)
                    if (mask.valid(i, j) && !(band.extra_diag && j == i)) {
                        lo = std::min(lo, j);
                        hi = std::max(hi, j + 1);
                    }
            }
            if (lo >= hi) lo = hi = 0;
            EXPECT_EQ(need[b].lo, lo) << c.name << " b=" << b;
            EXPECT_EQ(need[b].hi, hi) << c.name << " b=" << b;
        }

        const ha::kv_blocks_t blocks(p, need);
        std::vector<int64_t> jobs;
        int64_t g = 0;
        for (int64_t b = 0; b < p.num_seqs; ++b) {
            const int64_t len = offsets[b + 1] - offsets[b];
            EXPECT_EQ(blocks.first_block(b), g) << c.name;
            for (int64_t k0 = 0; k0 < len; k0 += ha::kBlockN, ++g) {
                const ha::kv_block_t blk = blocks.block(g);
                EXPECT_EQ(blk.g, g) << c.name;
                EXPECT_EQ(blk.b, b) << c.name;
                EXPECT_EQ(blk.k0, k0) << c.name;
                EXPECT_EQ(blk.nb, std::min(ha::kBlockN, len - k0)) << c.name;
                if (k0 < need[b].hi && k0 + ha::kBlockN > need[b].lo)
                    jobs.push_back(g);
            }
        }
        EXPECT_EQ(blocks.count(), g) << c.name;
        EXPECT_EQ(blocks.jobs(), jobs) << c.name;
    }
}

TEST(HstuAttentionPlan, BlockHeightGivesEveryThreadTiles) {
    for (const hstu_case &c : make_cases()) {
        std::vector<int64_t> offsets;
        const ha::hstu_attention_params p = plan_params(c, offsets);
        auto tiles = [&](int64_t bm) {
            int64_t n = 0;
            for (int64_t b = 0; b < p.num_seqs; ++b)
                n += (ha::seq_rows(p, b).num_q + bm - 1) / bm;
            return n * p.num_heads;
        };
        for (int32_t threads : {1, 5, 32, 64, 256}) {
            const int64_t bm = ha::choose_block_m(p, threads);
            EXPECT_TRUE(bm == 32 || bm == 64 || bm == 128) << bm;
            const int64_t want = ha::kTilesPerThread * threads;
            if (bm > ha::kMinBlockM) { EXPECT_GE(tiles(bm), want) << c.name; }
            if (bm < ha::kMaxBlockM) {
                EXPECT_LT(tiles(2 * bm), want) << c.name;
            }
        }
    }
}

/// The driver's per-tile windows: rows and bands from prepare_rows, the key
/// blocks it visits, block_window aligned to key_align and tight, and
/// chunk_bounds equal to the bands.
template <int64_t key_align>
void check_windows(const hstu_case &c) {
    std::vector<int64_t> offsets;
    const ha::hstu_attention_params p = plan_params(c, offsets);
    ha::arena_t arena;
    constexpr int64_t width = 64;
    for (int64_t bm : {32, 64, 128}) {
        const ha::tile_pool_t pool(1, bm, 0, 0, p.v_dim, arena);
        const ha::tile_t t = pool.get(0);
        const ha::work_plan_t plan(p, bm);
        for (int64_t z = 0; z < plan.size(); ++z) {
            const ha::work_item_t w = plan.item(z);
            const ha::key_range_t keys = ha::prepare_rows(p, w, t);
            const ha::seq_mask_t mask(p, w.s.len, w.s.targets);
            for (int64_t r = 0; r < w.rows; ++r) {
                const ha::row_band_t band = mask.band(w.s.q_pos + w.row0 + r);
                ASSERT_EQ(t.bands[r].lo, band.lo) << c.name;
                ASSERT_EQ(t.bands[r].hi, band.hi) << c.name;
                ASSERT_EQ(t.bands[r].extra_diag, band.extra_diag) << c.name;
            }
            for (int64_t k0 = 0; k0 < w.s.len; k0 += ha::kBlockN) {
                const int64_t nb = std::min(ha::kBlockN, w.s.len - k0);
                int64_t ra = w.rows, rb = 0, lo = nb, hi = 0;
                for (int64_t r = 0; r < w.rows; ++r) {
                    const int64_t a = std::max(t.bands[r].lo, k0) - k0;
                    const int64_t e = std::min(t.bands[r].hi, k0 + nb) - k0;
                    if (a < e) {
                        ra = std::min(ra, r);
                        rb = r + 1;
                        lo = std::min(lo, a);
                        hi = std::max(hi, e);
                    }
                }
                ha::block_window_t bw;
                const bool any = ha::block_window<key_align>(
                        t.bands, w.rows, 7, k0, nb, bw);
                ASSERT_EQ(any, ra < rb) << c.name << " z=" << z;
                if (!any) continue;
                EXPECT_GE(k0, keys.lo / ha::kBlockN * ha::kBlockN) << c.name;
                EXPECT_LT(k0, keys.hi) << c.name;
                EXPECT_EQ(bw.g, 7);
                EXPECT_EQ(bw.k0, k0);
                EXPECT_EQ(bw.nb, nb);
                EXPECT_EQ(bw.ra, ra) << c.name;
                EXPECT_EQ(bw.rb, rb) << c.name;
                EXPECT_EQ(bw.plo, lo / key_align * key_align) << c.name;
                EXPECT_EQ(bw.phi,
                        std::min(nb,
                                (hi + key_align - 1) / key_align * key_align))
                        << c.name;
                for (int64_t c0 = bw.plo / width * width; c0 < bw.phi;
                        c0 += width) {
                    ha::chunk_bounds(
                            t.bands, ra, rb, k0 + c0, width, t.lo, t.hi);
                    for (int64_t r = ra; r < rb; ++r)
                        for (int64_t j = 0; j < width; ++j) {
                            const int64_t key = k0 + c0 + j;
                            ASSERT_EQ(j >= t.lo[r] && j < t.hi[r],
                                    key >= t.bands[r].lo && key < t.bands[r].hi)
                                    << c.name << " r=" << r << " key=" << key;
                        }
                }
            }
        }
    }
}

TEST(HstuAttentionDriver, BlockWindowsCoverBands) {
    for (const hstu_case &c : make_cases()) {
        if (!c.causal) continue;
        check_windows<2>(c);
        check_windows<4>(c);
    }
}

TEST(HstuAttentionApi, RejectsInvalidArguments) {
    const hstu_case c {"invalid", {10, 20}};
    hstu_problem prob(c);
    std::vector<float> got;
    prob.offsets[1] = 30; // second sequence length becomes negative
    EXPECT_EQ(prob.run(kernel_t::reference, got), status_t::failure);
    prob.offsets[1] = 10;
    prob.max_len = 15; // shorter than the second sequence
    EXPECT_EQ(prob.run(kernel_t::reference, got), status_t::failure);
}

/// Valid parameters for two sequences of 10 and 20 tokens, H=2, D=V=8.
struct api_fixture {
    std::vector<bfloat16_t> q, k, v, out;
    std::vector<int64_t> offsets {0, 10, 30}, targets {2, 3};
    ha::hstu_attention_params p;
    api_fixture() : q(30 * 16), k(30 * 16), v(30 * 16), out(30 * 16) {
        p.num_seqs = 2;
        p.num_heads = 2;
        p.qk_dim = p.v_dim = 8;
        p.max_seq_len = 20;
        p.seq_offsets = offsets.data();
        p.num_targets = targets.data();
        p.q_stride_t = p.k_stride_t = p.v_stride_t = p.o_stride_t = 16;
        p.q_stride_h = p.k_stride_h = p.v_stride_h = p.o_stride_h = 8;
        p.kernel = kernel_t::reference;
    }
    status_t run(const void *qp = nullptr) {
        return ha::hstu_attention_direct(
                qp ? qp : q.data(), k.data(), v.data(), out.data(), p);
    }
};

TEST(HstuAttentionApi, RejectsInvalidParameters) {
    if (env_overrides_kernel()) GTEST_SKIP();
    {
        api_fixture f;
        EXPECT_EQ(f.run(), status_t::success);
    }
    auto expect_failure = [](auto mutate, const char *what) {
        api_fixture f;
        mutate(f);
        EXPECT_EQ(f.run(), status_t::failure) << what;
    };
    expect_failure([](api_fixture &f) { f.p.seq_offsets = nullptr; },
            "null seq_offsets");
    expect_failure([](api_fixture &f) { f.p.num_seqs = -1; }, "num_seqs < 0");
    expect_failure(
            [](api_fixture &f) { f.targets[1] = 21; }, "num_targets > length");
    expect_failure(
            [](api_fixture &f) { f.targets[0] = -1; }, "num_targets < 0");
    expect_failure([](api_fixture &f) { f.offsets = {0, 20, 10}; },
            "decreasing offsets");
    expect_failure([](api_fixture &f) { f.p.k_stride_t = 0; }, "zero stride");
    expect_failure([](api_fixture &f) { f.p.qkv_dt = data_type_t::s8; },
            "unsupported dtype");
    expect_failure([](api_fixture &f) { f.p.o_stride_h = 4; },
            "overlapping output heads");
    expect_failure([](api_fixture &f) { f.p.o_stride_t = 8; },
            "overlapping output rows");
    static const float scale[30] = {};
    expect_failure([](api_fixture &f) {
        f.p.delta_size = 4;
        f.p.attn_scale_per_token = scale;
    }, "per-token scale in delta mode");
    {
        api_fixture f;
        EXPECT_EQ(f.run(nullptr), status_t::success);
        f.p.num_seqs = 0;
        EXPECT_EQ(f.run(), status_t::success) << "no sequences";
    }
}

TEST(HstuAttentionApi, KeepsTheCallersThreadCount) {
    if (env_overrides_kernel()) GTEST_SKIP();
    const int before = omp_get_max_threads();
    for (kernel_t kernel : {kernel_t::reference, kernel_t::tiled}) {
        if (kernel == kernel_t::tiled && !has_bf16()) continue;
        for (int32_t nt : {0, 2}) {
            omp_set_num_threads(3);
            api_fixture f;
            f.p.kernel = kernel;
            f.p.num_threads = nt;
            EXPECT_EQ(f.run(), status_t::success);
            EXPECT_EQ(omp_get_max_threads(), 3)
                    << "kernel " << static_cast<int>(kernel) << " nt " << nt;
        }
    }
    omp_set_num_threads(before);
}

TEST(HstuAttentionApi, DeltaLongerThanSequenceFails) {
    hstu_case c {"delta_too_long", {10, 4}};
    c.delta = 5;
    const hstu_problem prob(c);
    std::vector<float> got;
    EXPECT_EQ(prob.run(kernel_t::reference, got), status_t::failure);
}

} // namespace
