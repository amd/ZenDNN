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

/// Cases, inputs and FP64 oracles shared by the HSTU attention gtests
/// (test_hstu_attention.cpp, test_hstu_attention_int8.cpp): the
/// generative-recommenders mask, `pytorch_hstu_mha` / `pytorch_cached_hstu_mha`
/// in FP64, and the same with the dynamic-INT8 kernel's quantization.

#ifndef HSTU_ATTENTION_TEST_UTILS_HPP
#define HSTU_ATTENTION_TEST_UTILS_HPP

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/hstu_attention/common/lowoha_hstu_mask.hpp"
#include "lowoha_operators/hstu_attention/lowoha_hstu_attention.hpp"
#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled.hpp"
#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_plan.hpp"

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

namespace hstu_test {

using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;
using zendnnl::common::status_t;
namespace ha = zendnnl::lowoha::hstu_attention;
using kernel_t = ha::hstu_attention_kernel_t;

inline bool has_bf16() {
    const auto &pi = zendnnl::common::zendnnl_platform_info();
    return pi.get_avx512_bf16_status() && pi.get_avx512_bw_vl_status();
}

/// `_get_valid_attn_mask` of generative-recommenders for one sequence;
/// `targets` < 0 means no num_targets tensor.
inline bool oracle_valid(int64_t n, int64_t targets, int64_t ctx,
        int64_t window, int64_t min_full, bool causal, int64_t i, int64_t j) {
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

/// Kernel selection and the INT8 flags can be forced through the
/// environment, which would change what these tests exercise.
inline bool env_overrides_kernel() {
    for (const char *name :
            {"ZENDNNL_HSTU_ATTENTION_KERNEL", "ZENDNNL_HSTU_ATTENTION_INT8_QK",
                    "ZENDNNL_HSTU_ATTENTION_INT8_PV"}) {
        const char *k = std::getenv(name);
        if (k && *k) return true;
    }
    return false;
}

inline bool has_int8() {
    return has_bf16() && ha::hstu_attention_tiled_int8_isa();
}

/// The INT8 kernel's quantization of a group with max |x| = amax: S8 values
/// RNE(x * quant_inv) of the exact product (the kernel rounds it once, in an
/// FMA), dequantized by quant_scale.
inline float quant_inv(float amax) {
    return 127.f / std::max(amax, 1e-10f);
}
inline float quant_scale(float amax) {
    return std::max(amax, 1e-10f) / 127.f;
}
inline float quant(float x, float inv) {
    return static_cast<float>(
            std::nearbyint(static_cast<double>(x) * static_cast<double>(inv)));
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

inline std::ostream &operator<<(std::ostream &os, const hstu_case &c) {
    return os << c.name;
}

inline std::vector<hstu_case> make_cases() {
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

    /**
     * FP64 output with the INT8 kernel's quantization (in FP32, as the
     * kernel quantizes). `qk_quant`: Q and K per (token, head), and the
     * dequantized score times alpha. `pv_quant`: V per (head, 256-key block
     * of the sequence, channel) and P = mask * silu per (row, block), with
     * scale(i) applied to the dequantized P*V of the block; otherwise
     * P = BF16(silu * scale), as the BF16 P*V reads it. The diagonal of a
     * row outside its band (a target row) stays exact.
     */
    std::vector<double> oracle_int8(bool qk_quant, bool pv_quant) const {
        const int64_t B = cs.lens.size(), H = cs.heads, D = cs.qk_dim,
                      Dv = cs.v_dim, NB = ha::kBlockN;
        std::vector<double> out(q_rows * H * Dv, 0.0);
        ha::hstu_attention_params mp;
        mp.causal = cs.causal;
        mp.max_attn_len = cs.window;
        mp.contextual_seq_len = cs.ctx;
        mp.min_full_attn_seq_len = cs.min_full;
        for (int64_t b = 0; b < B; ++b) {
            const int64_t n = cs.lens[b], base = offsets[b];
            const int64_t t = cs.targets.empty() ? -1 : cs.targets[b];
            const int64_t nq = cs.delta > 0 ? cs.delta : n;
            const int64_t pos0 = cs.delta > 0 ? n - cs.delta : 0;
            const int64_t qrow0 = cs.delta > 0 ? b * cs.delta : base;
            const ha::seq_mask_t mask(mp, n, t);
            for (int64_t h = 0; h < H; ++h) {
                std::vector<float> k8(n * D), ks(n), q8(D);
                for (int64_t j = 0; j < n; ++j) {
                    float am = 0.f;
                    for (int64_t d = 0; d < D; ++d)
                        am = std::max(am, std::abs(k_at(base + j, h, d)));
                    for (int64_t d = 0; d < D; ++d)
                        k8[j * D + d]
                                = quant(k_at(base + j, h, d), quant_inv(am));
                    ks[j] = quant_scale(am);
                }
                std::vector<float> v8(n * Dv), vs((n + NB - 1) / NB * Dv);
                for (int64_t k0 = 0; k0 < n; k0 += NB)
                    for (int64_t d = 0; d < Dv; ++d) {
                        const int64_t k1 = std::min(n, k0 + NB);
                        float am = 0.f;
                        for (int64_t j = k0; j < k1; ++j)
                            am = std::max(am, std::abs(v_at(base + j, h, d)));
                        for (int64_t j = k0; j < k1; ++j)
                            v8[j * Dv + d] = quant(
                                    v_at(base + j, h, d), quant_inv(am));
                        vs[k0 / NB * Dv + d] = quant_scale(am);
                    }
                for (int64_t r = 0; r < nq; ++r) {
                    const int64_t i = pos0 + r;
                    const double sc = cs.per_token_scale ? scale[base + i]
                            : cs.has_scale               ? cs.scale_value
                                           : 1.0 / static_cast<double>(max_len);
                    float am = 0.f;
                    for (int64_t d = 0; d < D; ++d)
                        am = std::max(am, std::abs(q_at(qrow0 + r, h, d)));
                    for (int64_t d = 0; d < D; ++d)
                        q8[d] = quant(q_at(qrow0 + r, h, d), quant_inv(am));
                    const double qs = quant_scale(am);
                    const ha::row_band_t band = mask.band(i);
                    double *o = &out[((qrow0 + r) * H + h) * Dv];
                    auto exact_score = [&](int64_t j) {
                        double s = 0.0;
                        for (int64_t d = 0; d < D; ++d)
                            s += double(q_at(qrow0 + r, h, d))
                                    * k_at(base + j, h, d);
                        return s;
                    };
                    std::vector<double> pb(NB);
                    for (int64_t k0 = 0; k0 < n; k0 += NB) {
                        const int64_t k1 = std::min(n, k0 + NB);
                        float pmax = 0.f;
                        for (int64_t j = k0; j < k1; ++j) {
                            pb[j - k0] = 0.0;
                            if (!oracle_valid(n, t, cs.ctx, cs.window,
                                        cs.min_full, cs.causal, i, j)
                                    || (band.extra_diag && j == i))
                                continue;
                            double s = 0.0;
                            if (qk_quant) {
                                for (int64_t d = 0; d < D; ++d)
                                    s += double(q8[d]) * k8[j * D + d];
                                s *= qs * ks[j];
                            } else {
                                s = exact_score(j);
                            }
                            const double x = alpha * s;
                            const double silu = x / (1.0 + std::exp(-x));
                            pb[j - k0] = silu;
                            pmax = std::max(pmax, std::abs(float(silu)));
                            if (pv_quant) continue;
                            const double w = static_cast<float>(
                                    bfloat16_t(float(silu * sc)));
                            for (int64_t d = 0; d < Dv; ++d)
                                o[d] += w * v_at(base + j, h, d);
                        }
                        if (!pv_quant) continue;
                        const float inv = quant_inv(pmax);
                        const double ps = quant_scale(pmax) * sc;
                        for (int64_t d = 0; d < Dv; ++d) {
                            double acc = 0.0;
                            for (int64_t j = k0; j < k1; ++j)
                                acc += double(quant(float(pb[j - k0]), inv))
                                        * v8[j * Dv + d];
                            o[d] += ps * vs[k0 / NB * Dv + d] * acc;
                        }
                    }
                    if (band.extra_diag) {
                        const double x = alpha * exact_score(i);
                        const double w = x / (1.0 + std::exp(-x)) * sc;
                        for (int64_t d = 0; d < Dv; ++d)
                            o[d] += w * v_at(base + i, h, d);
                    }
                }
            }
        }
        return out;
    }

    /// Runs the op; returns the output as float [q_rows, H, v_dim] (rows
    /// before first_row as 0) and whether every canary (rows before
    /// first_row, row padding) is untouched. `fast_silu` >= 0 calls the tiled
    /// kernel (INT8 with a quant flag) directly with that silu, bypassing the
    /// environment. `varlen` is params.is_varlen.
    status_t run(kernel_t kernel, std::vector<float> &result,
            int32_t num_threads = 0, bool *canary_ok = nullptr,
            bool qk_quant = false, bool pv_quant = false, int fast_silu = -1,
            bool varlen = true) const {
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
        p.is_qk_quant = qk_quant;
        p.is_pv_quant = pv_quant;
        p.kernel = kernel;
        p.is_varlen = varlen;

        const size_t out_elems = q_rows * o_stride_t;
        std::vector<float> out_f(out_elems, -7.f);
        std::vector<bfloat16_t> out_b(out_elems, bfloat16_t(-7.f));
        void *out = cs.out_dt == data_type_t::f32
                ? static_cast<void *>(out_f.data())
                : static_cast<void *>(out_b.data());
        const void *k = kv + k_off * esz, *v = kv + v_off * esz;
        status_t st;
        if (fast_silu < 0)
            st = ha::hstu_attention_direct(qbuf, k, v, out, p);
        else if (qk_quant || pv_quant)
            st = ha::hstu_attention_tiled_int8(
                    qbuf, k, v, out, p, fast_silu == 1);
        else
            st = ha::hstu_attention_tiled_bf16(
                    qbuf, k, v, out, p, fast_silu == 1);
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
inline double rel_error(
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

inline hstu_case find_case(const std::string &name) {
    for (const hstu_case &c : make_cases())
        if (c.name == name) return c;
    ADD_FAILURE() << "no case " << name;
    return {};
}

inline bool bitwise_equal(
        const std::vector<float> &a, const std::vector<float> &b) {
    return a.size() == b.size()
            && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

/// The tiled cases, plus scores large enough (|alpha q.k| beyond 100) that
/// exp(-x) saturates to 0 and inf.
inline std::vector<hstu_case> fast_silu_cases() {
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

} // namespace hstu_test

#endif // HSTU_ATTENTION_TEST_UTILS_HPP
