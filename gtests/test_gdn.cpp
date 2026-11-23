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

/// Gated DeltaNet (GDN): gdn_direct on decode-only, prefill-only and mixed
/// batches. The native kernels are checked against the FP32 reference kernel,
/// the reference prefill against repeated reference decode steps, and
/// prefills split over two calls against one call.

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"
#include "lowoha_operators/gdn/lowoha_gdn.hpp"
#include "lowoha_operators/gdn/lowoha_gdn_utils.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

namespace {

using zendnnl::common::data_type_t;
using zendnnl::common::status_t;
namespace ng = zendnnl::lowoha::gdn;

bool has_native_isa() {
    const auto &pi = zendnnl::common::zendnnl_platform_info();
    return pi.get_avx512f_status() && pi.get_avx512_bf16_status()
            && pi.get_avx512_bw_vl_status();
}

// ZENDNNL_GDN_KERNEL overrides the requested kernel, which would make both
// sides of a native-vs-reference comparison the same kernel, and
// ZENDNNL_GDN_PREFILL the prefill algorithm a tolerance is chosen for.
bool kernel_env_set() {
    return std::getenv("ZENDNNL_GDN_KERNEL") != nullptr;
}
bool prefill_env_set() {
    return std::getenv("ZENDNNL_GDN_PREFILL") != nullptr;
}

#define GDN_SKIP_UNLESS_NATIVE() \
    if (!has_native_isa()) { GTEST_SKIP() << "needs AVX-512 BF16"; } \
    if (kernel_env_set()) { GTEST_SKIP() << "ZENDNNL_GDN_KERNEL is set"; } \
    if (prefill_env_set()) { GTEST_SKIP() << "ZENDNNL_GDN_PREFILL is set"; }

// Invalid calls are rejected only while validation runs.
#define GDN_SKIP_UNLESS_VALIDATING() \
    if (!zendnnl::common::op_instrumentation::is_enabled()) { \
        GTEST_SKIP() << "ZENDNNL_DIAGNOSTICS_ENABLE=0 skips validation"; \
    }

float bf16_to_f32(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((bits >> 16) | 0x40u);
    }
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const int32_t exp = (h >> 10) & 0x1f;
    const uint32_t man = h & 0x3ffu;
    float mag;
    if (exp == 0) {
        mag = std::ldexp(static_cast<float>(man), -24);
    } else if (exp == 31) {
        mag = man ? std::numeric_limits<float>::quiet_NaN()
                  : std::numeric_limits<float>::infinity();
    } else {
        mag = std::ldexp(static_cast<float>(man | 0x400u), exp - 25);
    }
    uint32_t bits;
    std::memcpy(&bits, &mag, sizeof(bits));
    bits |= sign;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Round-to-nearest-even; test values stay in the normal f16 range.
uint16_t f32_to_f16(float f) {
    if (std::isnan(f)) { return 0x7e00u; }
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const float a = std::fabs(f);
    if (a < std::ldexp(1.0f, -14)) {
        return static_cast<uint16_t>(
                sign | static_cast<uint16_t>(std::nearbyint(a * 16777216.0f)));
    }
    int e;
    const float m = std::frexp(a, &e); // a = m * 2^e, m in [0.5, 1)
    uint32_t man = static_cast<uint32_t>(std::nearbyint(m * 2048.0f));
    if (man == 2048u) {
        man = 1024u;
        ++e;
    }
    return static_cast<uint16_t>(
            sign | static_cast<uint16_t>((e + 14) << 10) | (man - 1024u));
}

/// Typed buffer stored as raw 16-bit words (two per f32 element).
struct buf_t {
    data_type_t dt;
    std::vector<uint16_t> raw;

    buf_t() : dt(data_type_t::f32) {}
    buf_t(data_type_t d, size_t n)
        : dt(d), raw(n * (d == data_type_t::f32 ? 2 : 1), 0) {}

    size_t size() const {
        return dt == data_type_t::f32 ? raw.size() / 2 : raw.size();
    }
    void *data() { return raw.data(); }
    const void *data() const { return raw.data(); }

    float get(size_t i) const {
        if (dt == data_type_t::f32) {
            float f;
            std::memcpy(&f, raw.data() + 2 * i, sizeof(f));
            return f;
        }
        return dt == data_type_t::f16 ? f16_to_f32(raw[i])
                                      : bf16_to_f32(raw[i]);
    }
    void set(size_t i, float v) {
        if (dt == data_type_t::f32) {
            std::memcpy(raw.data() + 2 * i, &v, sizeof(v));
        } else {
            raw[i] = dt == data_type_t::f16 ? f32_to_f16(v) : f32_to_bf16(v);
        }
    }
    void fill_uniform(std::mt19937 &rng, float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        for (size_t i = 0; i < size(); ++i) {
            set(i, d(rng));
        }
    }
};

// Distinct, shuffled state slots: sequences must never share a slot.
std::vector<int32_t> distinct_slots(int64_t n, int64_t slots, unsigned seed) {
    std::vector<int32_t> all(slots);
    for (int64_t i = 0; i < slots; ++i) {
        all[i] = static_cast<int32_t>(i);
    }
    std::mt19937 rng(seed);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(n);
    return all;
}

struct shape_t {
    int64_t H, HV, K, V, W;
};

std::ostream &operator<<(std::ostream &os, const shape_t &s) {
    return os << "H" << s.H << "_HV" << s.HV << "_K" << s.K << "_V" << s.V
              << "_W" << s.W;
}

/// All the tensors of one GDN call plus the params that describe them.
struct problem_t {
    shape_t s;
    int64_t tokens;
    int64_t slots;
    ng::gdn_params p;
    buf_t qkv, a, b, a_log, dt_bias, conv_w, conv_b, conv_state, ssm_state, out,
            z, norm_w;
    std::vector<int32_t> idx;

    problem_t(const shape_t &sh, int64_t tokens_, int64_t slots_,
            data_type_t state_dt, ng::gdn_state_layout_t layout, bool fuse_norm,
            uint32_t seed, data_type_t src = data_type_t::bf16)
        : s(sh), tokens(tokens_), slots(slots_) {
        std::mt19937 rng(seed);
        const int64_t D = 2 * s.H * s.K + s.HV * s.V;
        // Padded rows, like the strided views vLLM hands over.
        const int64_t qkv_stride = D + 16;
        const int64_t ab_stride = 2 * s.HV;

        qkv = buf_t(src, tokens * qkv_stride);
        a = buf_t(src, tokens * ab_stride);
        b = buf_t(src, tokens * ab_stride);
        a_log = buf_t(data_type_t::f32, s.HV);
        dt_bias = buf_t(src, s.HV);
        conv_w = buf_t(src, D * s.W);
        conv_b = buf_t(src, D);
        conv_state = buf_t(src, slots * (s.W > 0 ? s.W - 1 : 0) * D + 1);
        ssm_state = buf_t(state_dt, slots * s.HV * s.K * s.V);
        out = buf_t(src, tokens * s.HV * s.V);
        z = buf_t(src, tokens * s.HV * s.V);
        norm_w = buf_t(data_type_t::f32, s.V);

        qkv.fill_uniform(rng, -1.0f, 1.0f);
        a.fill_uniform(rng, -2.0f, 2.0f);
        b.fill_uniform(rng, -3.0f, 3.0f);
        a_log.fill_uniform(rng, -1.0f, 1.0f);
        dt_bias.fill_uniform(rng, -0.5f, 0.5f);
        conv_w.fill_uniform(rng, -0.5f, 0.5f);
        conv_b.fill_uniform(rng, -0.1f, 0.1f);
        conv_state.fill_uniform(rng, -1.0f, 1.0f);
        ssm_state.fill_uniform(rng, -0.5f, 0.5f);
        z.fill_uniform(rng, -2.0f, 2.0f);
        norm_w.fill_uniform(rng, 0.5f, 1.5f);
        // NaN canary: an output row a kernel should write but does not shows
        // up as a mismatch against the reference.
        for (size_t i = 0; i < out.size(); ++i) {
            out.set(i, std::numeric_limits<float>::quiet_NaN());
        }

        p.batch = tokens;
        p.num_tokens = tokens;
        p.num_k_heads = s.H;
        p.num_v_heads = s.HV;
        p.head_k_dim = s.K;
        p.head_v_dim = s.V;
        p.conv_width = s.W;
        p.silu = true;
        p.qkv_stride = qkv_stride;
        p.a_stride = ab_stride;
        p.b_stride = ab_stride;
        p.out_stride_t = s.HV * s.V;
        p.out_stride_h = s.V;
        // SD conv state: [slot][tap][channel].
        p.conv_state_slot_stride = (s.W > 0 ? s.W - 1 : 0) * D;
        p.conv_state_tap_stride = D;
        p.conv_state_dim_stride = 1;
        p.ssm_state_slot_stride = s.HV * s.K * s.V;
        p.ssm_state_head_stride = s.K * s.V;
        p.state_layout = layout;
        p.src_dt = src;
        p.state_dt = state_dt;
        p.a_log_dt = data_type_t::f32;
        p.dt_bias_dt = src;
        if (fuse_norm) {
            p.norm_weight = norm_w.data();
            p.norm_weight_dt = data_type_t::f32;
            p.z = z.data();
            p.z_stride_t = s.HV * s.V;
            p.z_stride_h = s.V;
        }
    }

    // b lives in the second half of each ab row, as after ba.chunk(2).
    const void *b_ptr() const {
        return reinterpret_cast<const uint16_t *>(b.data()) + s.HV;
    }

    // gdn_direct with q's batch description on this problem's tensors.
    status_t call(ng::gdn_params q, ng::gdn_kernel_t kernel) {
        q.kernel = kernel;
        if (q.norm_weight) {
            q.norm_weight = norm_w.data();
            q.z = z.data();
        }
        return ng::gdn_direct(qkv.data(), a.data(), b_ptr(), a_log.data(),
                dt_bias.data(), conv_w.data(), conv_b.data(), conv_state.data(),
                ssm_state.data(), idx.data(), out.data(), q);
    }

    // p.batch decodes, one token each (no cu_seqlens).
    status_t decode(ng::gdn_kernel_t kernel) {
        ng::gdn_params q = p;
        q.num_decodes = q.batch;
        q.num_tokens = q.batch;
        return call(q, kernel);
    }

    // DS conv state: [slot][channel][tap], as vLLM's transposed view.
    void use_ds_conv_state() {
        const int64_t taps = s.W > 0 ? s.W - 1 : 0;
        p.conv_state_tap_stride = 1;
        p.conv_state_dim_stride = taps;
    }

    // One prefill per entry of idx over the tokens rows that qsl splits.
    status_t prefill(ng::gdn_kernel_t kernel, const std::vector<int32_t> &qsl,
            const std::vector<uint8_t> &init, bool null_init = false) {
        return mixed(kernel, 0, qsl, init, null_init);
    }

    // The first nd entries of idx decode, the rest prefill; an empty qsl
    // passes a null cu_seqlens (one token per sequence).
    status_t mixed(ng::gdn_kernel_t kernel, int64_t nd,
            const std::vector<int32_t> &qsl, const std::vector<uint8_t> &init,
            bool null_init = false) {
        ng::gdn_params q = p;
        q.batch = static_cast<int64_t>(idx.size());
        q.num_decodes = nd;
        q.num_tokens = qsl.empty() ? q.batch : tokens;
        q.cu_seqlens = qsl.empty() ? nullptr : qsl.data();
        q.has_initial_state = null_init
                ? nullptr
                : reinterpret_cast<const bool *>(init.data());
        return call(q, kernel);
    }
};

void expect_close(const buf_t &x, const buf_t &y, float atol, float rtol,
        const char *what) {
    ASSERT_EQ(x.size(), y.size());
    size_t bad = 0;
    float worst = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        // Rows both kernels leave untouched keep the NaN canary.
        if (std::isnan(x.get(i)) && std::isnan(y.get(i))) { continue; }
        const float d = std::fabs(x.get(i) - y.get(i));
        const float tol = atol + rtol * std::fabs(y.get(i));
        if (!(d <= tol)) {
            ++bad;
            worst = std::max(worst, d);
        }
    }
    EXPECT_EQ(bad, 0u) << what << ": " << bad << " of " << x.size()
                       << " elements out of tolerance, worst abs diff "
                       << worst;
}

const char *dt_tag(data_type_t dt) {
    return dt == data_type_t::f32    ? "f32"
            : dt == data_type_t::f16 ? "f16"
                                     : "bf16";
}

// The state type in a test name, and the source type when it is not bf16.
std::string types_tag(data_type_t src, data_type_t state_dt) {
    return std::string(dt_tag(state_dt)) + "state"
            + (src == data_type_t::bf16
                            ? ""
                            : std::string("_") + dt_tag(src) + "src");
}

// Native-vs-reference tolerances. Both kernels round the output to the
// source type, and they sum in different orders, so an output can differ by
// one or two ulps of src; a state by FP32 summation noise plus an ulp of the
// state type.
struct tol_t {
    float out, state_atol, state_rtol;
};

tol_t native_tol(data_type_t src, data_type_t state_dt) {
    if (src == data_type_t::bf16) { return {2e-2f, 2e-3f, 1e-2f}; }
    const float out = src == data_type_t::f16 ? 3e-3f : 1e-4f;
    if (state_dt == data_type_t::bf16) { return {out, 2e-3f, 1e-2f}; }
    if (state_dt == data_type_t::f16) { return {out, 5e-4f, 2e-3f}; }
    return {out, 1e-4f, 1e-4f};
}

// The chunked form multiplies bf16 operands for bf16 sources and FP32 ones
// otherwise (FP32 accumulation, decays and state): bf16 sources match the
// FP32 reference to bf16-level tolerances, f16 and f32 sources like the
// recurrent form, f16 up to its output's rounding. bf16 operands would fail
// the f16 and f32 tolerances.
tol_t chunked_tol(data_type_t src, data_type_t state_dt, bool fuse_norm) {
    tol_t t {};
    if (src == data_type_t::bf16) {
        // The norm brings outputs to unit scale, a few bf16 ulps above 3e-2.
        t = {fuse_norm ? 5e-2f : 3e-2f, 2e-2f, 2e-2f};
    } else if (src == data_type_t::f16) {
        // The norm divides by the row's RMS, which scales up f16 rounding on
        // rows of small norm.
        t = {fuse_norm ? 6e-3f : 1e-3f, 1e-4f, 1e-4f};
    } else {
        t = {1e-4f, 1e-4f, 1e-4f};
    }
    const tol_t r = native_tol(src, state_dt);
    t.state_atol = std::max(t.state_atol, r.state_atol);
    t.state_rtol = std::max(t.state_rtol, r.state_rtol);
    return t;
}

void expect_tol(problem_t &nat, problem_t &ref, const tol_t &t) {
    expect_close(nat.out, ref.out, t.out, t.out, "output");
    expect_close(nat.ssm_state, ref.ssm_state, t.state_atol, t.state_rtol,
            "ssm_state");
    expect_close(nat.conv_state, ref.conv_state, 0.0f, 0.0f, "conv_state");
}

struct decode_case_t {
    shape_t s;
    int64_t batch;
    data_type_t state_dt;
    ng::gdn_state_layout_t layout;
    bool fuse_norm;
    data_type_t src = data_type_t::bf16;
};

std::string decode_name(const testing::TestParamInfo<decode_case_t> &info) {
    const decode_case_t &c = info.param;
    std::ostringstream os;
    os << c.s << "_B" << c.batch << "_" << types_tag(c.src, c.state_dt) << "_"
       << (c.layout == ng::gdn_state_layout_t::kv ? "kv" : "vk")
       << (c.fuse_norm ? "_norm" : "");
    return os.str();
}

class GdnDecode : public testing::TestWithParam<decode_case_t> {};

TEST_P(GdnDecode, NativeMatchesReference) {
    GDN_SKIP_UNLESS_NATIVE();
    const decode_case_t &c = GetParam();
    const int64_t slots = c.batch + 3;
    problem_t ref(
            c.s, c.batch, slots, c.state_dt, c.layout, c.fuse_norm, 7, c.src);
    // Shuffled slots plus one padded sequence, whose z row nothing may read.
    ref.idx = distinct_slots(c.batch, slots, 5);
    if (c.batch > 2) {
        ref.idx[1] = -1;
        const int64_t row = c.s.HV * c.s.V;
        for (int64_t j = row; j < 2 * row; ++j) {
            ref.z.set(j, std::numeric_limits<float>::quiet_NaN());
        }
    }
    problem_t nat = ref;

    // Several steps so state errors would compound.
    for (int step = 0; step < 3; ++step) {
        ASSERT_EQ(ref.decode(ng::gdn_kernel_t::reference), status_t::success);
        ASSERT_EQ(nat.decode(ng::gdn_kernel_t::native), status_t::success);
        expect_tol(nat, ref, native_tol(c.src, c.state_dt));
    }
}

const shape_t kQwen36 {16, 32, 128, 128, 4};
const shape_t kQwen38 {16, 48, 128, 128, 4};
const shape_t kSmall {2, 4, 64, 32, 4};
const shape_t kNoConv {2, 2, 32, 48, 0};

constexpr auto kKv = ng::gdn_state_layout_t::kv;
constexpr auto kVk = ng::gdn_state_layout_t::vk;
constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kF16 = data_type_t::f16;

INSTANTIATE_TEST_SUITE_P(Gdn, GdnDecode,
        testing::Values(decode_case_t {kQwen36, 1, kF32, kKv, false},
                decode_case_t {kQwen36, 8, kF32, kKv, false},
                decode_case_t {kQwen36, 5, kBf16, kKv, false},
                decode_case_t {kQwen36, 4, kF32, kVk, false},
                decode_case_t {kQwen36, 6, kF32, kKv, true},
                decode_case_t {kSmall, 3, kF32, kKv, false},
                decode_case_t {kSmall, 7, kBf16, kVk, true},
                decode_case_t {kNoConv, 4, kF32, kKv, false},
                // f16 and f32 sources on every state type.
                decode_case_t {kQwen36, 8, kF32, kKv, false, kF16},
                decode_case_t {kQwen36, 5, kBf16, kKv, true, kF16},
                decode_case_t {kQwen36, 6, kF16, kKv, false, kF16},
                decode_case_t {kQwen36, 4, kF32, kVk, true, kF16},
                decode_case_t {kSmall, 7, kF16, kVk, false, kF16},
                decode_case_t {kQwen36, 8, kF32, kKv, true, kF32},
                decode_case_t {kSmall, 5, kBf16, kVk, false, kF32},
                decode_case_t {kSmall, 3, kF16, kKv, false, kF32}),
        decode_name);

struct prefill_case_t {
    shape_t s;
    std::vector<int32_t> lens;
    data_type_t state_dt;
    ng::gdn_state_layout_t layout;
    bool fuse_norm;
    data_type_t src = data_type_t::bf16;
};

std::string prefill_name(const testing::TestParamInfo<prefill_case_t> &info) {
    const prefill_case_t &c = info.param;
    std::ostringstream os;
    os << c.s << "_N" << c.lens.size() << "_" << types_tag(c.src, c.state_dt)
       << "_" << (c.layout == ng::gdn_state_layout_t::kv ? "kv" : "vk")
       << (c.fuse_norm ? "_norm" : "") << "_" << info.index;
    return os.str();
}

class GdnPrefill : public testing::TestWithParam<prefill_case_t> {};

TEST_P(GdnPrefill, NativeMatchesReference) {
    GDN_SKIP_UNLESS_NATIVE();
    const prefill_case_t &c = GetParam();
    std::vector<int32_t> qsl {0};
    for (int32_t l : c.lens) {
        qsl.push_back(qsl.back() + l);
    }
    const int64_t n = static_cast<int64_t>(c.lens.size());
    const int64_t slots = n + 2;
    problem_t ref(c.s, qsl.back(), slots, c.state_dt, c.layout, c.fuse_norm, 11,
            c.src);
    ref.idx = distinct_slots(n, slots, 3);
    std::vector<uint8_t> init(n);
    for (int64_t i = 0; i < n; ++i) {
        init[i] = static_cast<uint8_t>(i % 2);
    }
    problem_t nat = ref;
    nat.p.prefill_algo = ng::gdn_prefill_algo_t::recurrent;
    ASSERT_EQ(ref.prefill(ng::gdn_kernel_t::reference, qsl, init),
            status_t::success);
    ASSERT_EQ(nat.prefill(ng::gdn_kernel_t::native, qsl, init),
            status_t::success);
    expect_tol(nat, ref, native_tol(c.src, c.state_dt));
}

INSTANTIATE_TEST_SUITE_P(Gdn, GdnPrefill,
        testing::Values(prefill_case_t {kQwen36, {37, 1, 70}, kF32, kKv, false},
                prefill_case_t {kQwen36, {130}, kF32, kKv, false},
                prefill_case_t {kQwen36, {2, 65}, kBf16, kVk, false},
                prefill_case_t {kQwen36, {9, 20}, kF32, kKv, true},
                prefill_case_t {kSmall, {5, 1, 2, 64}, kF32, kKv, false},
                prefill_case_t {kNoConv, {3, 17}, kF32, kKv, false},
                // f16 and f32 sources on every state type.
                prefill_case_t {kQwen36, {37, 1, 70}, kF32, kKv, false, kF16},
                prefill_case_t {kQwen36, {2, 65}, kF16, kVk, false, kF16},
                prefill_case_t {kQwen36, {9, 20}, kBf16, kKv, true, kF16},
                prefill_case_t {kSmall, {5, 1, 2, 64}, kF16, kKv, true, kF16},
                prefill_case_t {kQwen36, {33, 4}, kF32, kKv, true, kF32},
                prefill_case_t {kSmall, {5, 1, 2, 64}, kF16, kVk, false, kF32},
                prefill_case_t {kNoConv, {3, 17}, kBf16, kKv, false, kF32}),
        prefill_name);

// The reference prefill of an L-token sequence must equal L reference decode
// steps: this pins the conv-state and recurrence semantics independently of
// the native kernels.
TEST(GdnReference, PrefillEqualsRepeatedDecode) {
    const shape_t s = kSmall;
    const int64_t L = 9;
    problem_t pre(
            s, L, 2, data_type_t::f32, ng::gdn_state_layout_t::kv, false, 3);
    pre.idx = {1};
    problem_t dec = pre;
    ASSERT_EQ(pre.prefill(ng::gdn_kernel_t::reference, {0, int32_t(L)}, {1}),
            status_t::success);

    const int64_t row = dec.p.qkv_stride;
    const int64_t abr = dec.p.a_stride;
    for (int64_t t = 0; t < L; ++t) {
        ng::gdn_params q = dec.p;
        q.batch = 1;
        q.num_decodes = 1;
        q.num_tokens = 1;
        q.kernel = ng::gdn_kernel_t::reference;
        const uint16_t *qkv = static_cast<const uint16_t *>(dec.qkv.data());
        const uint16_t *a = static_cast<const uint16_t *>(dec.a.data());
        const uint16_t *b = static_cast<const uint16_t *>(dec.b_ptr());
        uint16_t *out = static_cast<uint16_t *>(dec.out.data());
        ASSERT_EQ(ng::gdn_direct(qkv + t * row, a + t * abr, b + t * abr,
                          dec.a_log.data(), dec.dt_bias.data(),
                          dec.conv_w.data(), dec.conv_b.data(),
                          dec.conv_state.data(), dec.ssm_state.data(),
                          dec.idx.data(), out + t * q.out_stride_t, q),
                status_t::success);
    }
    expect_close(pre.out, dec.out, 1e-6f, 0.0f, "output");
    expect_close(pre.ssm_state, dec.ssm_state, 1e-6f, 0.0f, "ssm_state");
    expect_close(pre.conv_state, dec.conv_state, 0.0f, 0.0f, "conv_state");
}

std::vector<int32_t> offsets(const std::vector<int32_t> &lens) {
    std::vector<int32_t> qsl {0};
    for (int32_t l : lens) {
        qsl.push_back(qsl.back() + l);
    }
    return qsl;
}

void expect_same(problem_t &x, problem_t &y, float otol, const char *what) {
    SCOPED_TRACE(what);
    expect_close(x.out, y.out, otol, otol, "output");
    expect_close(x.ssm_state, y.ssm_state, 2e-3f, 1e-2f, "ssm_state");
    expect_close(x.conv_state, y.conv_state, 0.0f, 0.0f, "conv_state");
}

// x's conv state laid out with the given tap and channel strides over taps
// taps, slots one after another, refilled at random.
void set_conv_strides(problem_t &x, int64_t taps, int64_t tap_stride,
        int64_t dim_stride, uint32_t seed) {
    const int64_t D = 2 * x.s.H * x.s.K + x.s.HV * x.s.V;
    const int64_t slot = (taps - 1) * tap_stride + (D - 1) * dim_stride + 1;
    x.p.conv_state_tap_stride = tap_stride;
    x.p.conv_state_dim_stride = dim_stride;
    x.p.conv_state_slot_stride = slot;
    x.conv_state
            = buf_t(x.conv_state.dt, static_cast<size_t>(x.slots * slot + 1));
    std::mt19937 rng(seed);
    x.conv_state.fill_uniform(rng, -1.0f, 1.0f);
}

// Decode then prefill on both kernels with one params tweak applied.
void check_variant(const shape_t &s, data_type_t src, data_type_t state_dt,
        ng::gdn_state_layout_t layout, bool norm, void (*tweak)(problem_t &)) {
    problem_t ref(s, 6, 9, state_dt, layout, norm, 21, src);
    ref.idx = distinct_slots(6, 9, 2);
    ref.idx[2] = -1;
    if (tweak) { tweak(ref); }
    problem_t nat = ref;
    const tol_t tol = native_tol(src, state_dt);
    for (int step = 0; step < 2; ++step) {
        ASSERT_EQ(ref.decode(ng::gdn_kernel_t::reference), status_t::success);
        ASSERT_EQ(nat.decode(ng::gdn_kernel_t::native), status_t::success);
        SCOPED_TRACE("decode");
        expect_tol(nat, ref, tol);
    }
    const std::vector<int32_t> qsl = offsets({3, 1, 0, 2});
    std::vector<uint8_t> init {1, 0, 1, 1};
    ref.idx.resize(4);
    nat.idx = ref.idx;
    ASSERT_EQ(ref.prefill(ng::gdn_kernel_t::reference, qsl, init),
            status_t::success);
    ASSERT_EQ(nat.prefill(ng::gdn_kernel_t::native, qsl, init),
            status_t::success);
    SCOPED_TRACE("prefill");
    expect_tol(nat, ref, tol);
}

TEST(GdnVariants, SourceAndStateTypes) {
    GDN_SKIP_UNLESS_NATIVE();
    const data_type_t types[]
            = {data_type_t::f32, data_type_t::bf16, data_type_t::f16};
    for (data_type_t src : types) {
        for (data_type_t st : types) {
            SCOPED_TRACE(testing::Message()
                    << "src " << static_cast<int>(src) << " state "
                    << static_cast<int>(st));
            check_variant(kSmall, src, st, ng::gdn_state_layout_t::kv, false,
                    nullptr);
        }
    }
}

TEST(GdnVariants, Options) {
    GDN_SKIP_UNLESS_NATIVE();
    using tweak_t = void (*)(problem_t &);
    const tweak_t tweaks[] = {
            [](problem_t &x) { x.p.use_qk_l2norm = false; },
            [](problem_t &x) { x.p.silu = false; },
            [](problem_t &x) { x.p.round_conv_out = false; },
            [](problem_t &x) { x.p.scale = 0.3f; },
            [](problem_t &x) { x.use_ds_conv_state(); },
            [](problem_t &x) { x.p.num_threads = 1; },
            [](problem_t &x) { x.p.num_threads = 3; },
            [](problem_t &x) { x.p.num_threads = 64; },
            [](problem_t &x) {
        // Padded output and z rows.
        x.out = buf_t(x.out.dt, x.tokens * (x.s.HV * x.s.V + 32));
        for (size_t i = 0; i < x.out.size(); ++i) {
            x.out.set(i, std::numeric_limits<float>::quiet_NaN());
        }
        x.p.out_stride_t = x.s.HV * x.s.V + 32;
    },
    };
    for (size_t i = 0; i < sizeof(tweaks) / sizeof(tweaks[0]); ++i) {
        SCOPED_TRACE(testing::Message() << "tweak " << i);
        check_variant(kSmall, data_type_t::bf16, data_type_t::f32,
                ng::gdn_state_layout_t::kv, false, tweaks[i]);
    }
    SCOPED_TRACE("ds conv state, vk + norm");
    check_variant(kSmall, data_type_t::bf16, data_type_t::bf16,
            ng::gdn_state_layout_t::vk, true,
            [](problem_t &x) { x.use_ds_conv_state(); });
    for (data_type_t src : {data_type_t::f16, data_type_t::f32}) {
        SCOPED_TRACE(testing::Message() << "ds conv state, " << dt_tag(src));
        check_variant(kQwen36, src, data_type_t::f16,
                ng::gdn_state_layout_t::kv, false,
                [](problem_t &x) { x.use_ds_conv_state(); });
    }
}

// Conv states the API takes beyond vLLM's two layouts: DS with padding after
// each channel's taps, which decode reads as it reads DS, and channels at a
// stride of 2 in an SD-like layout, which it reads in scalar arithmetic.
// Decode and prefill match the reference, the conv state exactly.
TEST(GdnVariants, PaddedConvStateLayouts) {
    GDN_SKIP_UNLESS_NATIVE();
    using tweak_t = void (*)(problem_t &);
    const tweak_t tweaks[] = {[](problem_t &x) {
        set_conv_strides(x, x.s.W - 1, 1, x.s.W + 1, 151);
    }, [](problem_t &x) {
        const int64_t D = 2 * x.s.H * x.s.K + x.s.HV * x.s.V;
        set_conv_strides(x, x.s.W - 1, 2 * D, 2, 153);
    }};
    for (data_type_t src : {kBf16, kF32}) {
        for (size_t i = 0; i < 2; ++i) {
            SCOPED_TRACE(testing::Message()
                    << dt_tag(src)
                    << (i == 0 ? " padded DS" : " SD, channel stride 2"));
            check_variant(kSmall, src, kF32, kKv, false, tweaks[i]);
        }
    }
}

// The fused gated RMSNorm on the rounded (default) and the FP32 core output,
// with both gates, for bf16 and f16 sources (an f16 core output rounds to
// f16 before the norm). The Qwen3.6 shape has four value tiles per head,
// which the prefill norm pass over rounded rows keeps.
TEST(GdnVariants, FusedNorm) {
    GDN_SKIP_UNLESS_NATIVE();
    using tweak_t = void (*)(problem_t &);
    const tweak_t tweaks[] = {
            [](problem_t &) {},
            [](problem_t &x) { x.p.round_core_out = false; },
            [](problem_t &x) { x.p.norm_gate = ng::gdn_norm_gate_t::sigmoid; },
            [](problem_t &x) {
        x.p.round_core_out = false;
        x.p.norm_gate = ng::gdn_norm_gate_t::sigmoid;
    },
            [](problem_t &x) {
        // An f16 norm weight, as an fp16 model holds it.
        x.norm_w = buf_t(data_type_t::f16, x.s.V);
        std::mt19937 rng(3);
        x.norm_w.fill_uniform(rng, 0.5f, 1.5f);
        x.p.norm_weight_dt = data_type_t::f16;
    },
    };
    for (data_type_t src : {data_type_t::bf16, data_type_t::f16}) {
        for (size_t i = 0; i < sizeof(tweaks) / sizeof(tweaks[0]); ++i) {
            SCOPED_TRACE(testing::Message()
                    << dt_tag(src) << " source, tweak " << i);
            check_variant(kQwen36, src, data_type_t::f32,
                    ng::gdn_state_layout_t::kv, true, tweaks[i]);
        }
    }
}

// allow_neg_eigval doubles beta (OLMo's hybrid layers): decode, the recurrent
// prefill and the chunked prefill match the reference kernel, which doubles
// it too, and the flag changes the result.
TEST(GdnVariants, NegativeEigenvalueGate) {
    GDN_SKIP_UNLESS_NATIVE();
    const auto neg = [](problem_t &x) { x.p.allow_neg_eigval = true; };
    for (data_type_t src : {kBf16, kF32}) {
        for (bool norm : {false, true}) {
            SCOPED_TRACE(
                    testing::Message() << dt_tag(src) << (norm ? " norm" : ""));
            check_variant(kQwen36, src, kF32, kKv, norm, neg);
            const std::vector<int32_t> qsl = offsets({70, 9, 33});
            const std::vector<uint8_t> init {1, 0, 1};
            problem_t ref(kQwen36, qsl.back(), 5, kF32, kKv, norm, 131, src);
            ref.idx = distinct_slots(3, 5, 31);
            problem_t off = ref;
            neg(ref);
            problem_t nat = ref;
            nat.p.prefill_algo = ng::gdn_prefill_algo_t::chunked;
            ASSERT_EQ(ref.prefill(ng::gdn_kernel_t::reference, qsl, init),
                    status_t::success);
            ASSERT_EQ(nat.prefill(ng::gdn_kernel_t::native, qsl, init),
                    status_t::success);
            ASSERT_EQ(off.prefill(ng::gdn_kernel_t::reference, qsl, init),
                    status_t::success);
            SCOPED_TRACE("chunked prefill");
            // beta up to 2 doubles the weight of every rank-1 update, and with
            // it the rounding of bf16 operands, which the norm carries into
            // rows of small RMS.
            tol_t t = chunked_tol(src, kF32, norm);
            if (src == kBf16 && norm) { t.out *= 2.0f; }
            expect_tol(nat, ref, t);
            size_t changed = 0;
            for (size_t i = 0; i < ref.out.raw.size(); ++i) {
                changed += ref.out.raw[i] != off.out.raw[i];
            }
            EXPECT_GT(changed, 0u)
                    << "allow_neg_eigval left the outputs as they were";
        }
    }
}

struct chunked_case_t {
    shape_t s;
    std::vector<int32_t> lens;
    ng::gdn_state_layout_t layout;
    bool fuse_norm;
    data_type_t src = data_type_t::bf16;
    data_type_t state_dt = data_type_t::f32;
    // Threads of the native call (0: the OpenMP default) and the split they
    // must give (see chunk_split; null: not checked).
    int32_t threads = 0;
    const char *split = nullptr;
};

std::string chunked_name(const testing::TestParamInfo<chunked_case_t> &info) {
    const chunked_case_t &c = info.param;
    std::ostringstream os;
    os << c.s << "_N" << c.lens.size() << "_"
       << (c.layout == ng::gdn_state_layout_t::kv ? "kv" : "vk")
       << (c.fuse_norm ? "_norm" : "");
    if (c.src != data_type_t::bf16 || c.state_dt != data_type_t::f32) {
        os << "_" << types_tag(c.src, c.state_dt);
    }
    if (c.threads > 0) { os << "_T" << c.threads; }
    os << "_" << info.index;
    return os.str();
}

// The split of the chunked prefill of sequences [s0, s1) of a batch on
// `threads` threads: "1" for whole heads, else "<blocks>x<n>" with the last
// n (sequence, value head) pairs in blocks.
std::string chunk_split(const ng::gdn_params &p, const int32_t *slots,
        const std::vector<int32_t> &qsl, int64_t s0, int64_t s1,
        int32_t threads) {
    const ng::gdn_chunk_split_t s
            = ng::gdn_chunk_split(p, slots, qsl.data(), s0, s1, threads);
    const int64_t pairs = (s1 - s0) * p.num_v_heads;
    const int64_t n = s.blocks > 1 ? pairs - std::min(s.first_split, pairs) : 0;
    return n == 0 ? "1" : std::to_string(s.blocks) + "x" + std::to_string(n);
}

class GdnChunkedPrefill : public testing::TestWithParam<chunked_case_t> {};

// The chunked form against the FP32 reference at the precision of its GEMM
// operands (see chunked_tol); with f16 and f32 sources the tolerances are
// below what bf16 operands reach. Cases with many threads per (sequence,
// value head) split the heads into value blocks.
TEST_P(GdnChunkedPrefill, MatchesReference) {
    GDN_SKIP_UNLESS_NATIVE();
    const chunked_case_t &c = GetParam();
    const std::vector<int32_t> qsl = offsets(c.lens);
    const int64_t n = static_cast<int64_t>(c.lens.size());
    problem_t ref(c.s, qsl.back(), n + 2, c.state_dt, c.layout, c.fuse_norm, 17,
            c.src);
    ref.idx = distinct_slots(n, n + 2, 4);
    std::vector<uint8_t> init(n);
    for (int64_t i = 0; i < n; ++i) {
        init[i] = static_cast<uint8_t>(i % 2 == 0);
    }
    problem_t nat = ref;
    nat.p.prefill_algo = ng::gdn_prefill_algo_t::chunked;
    nat.p.num_threads = c.threads;
    if (c.split != nullptr) {
        ASSERT_EQ(chunk_split(nat.p, nat.idx.data(), qsl, 0, n, c.threads),
                c.split);
    }
    ASSERT_EQ(ref.prefill(ng::gdn_kernel_t::reference, qsl, init),
            status_t::success);
    ASSERT_EQ(nat.prefill(ng::gdn_kernel_t::native, qsl, init),
            status_t::success);
    expect_tol(nat, ref, chunked_tol(c.src, c.state_dt, c.fuse_norm));
}

INSTANTIATE_TEST_SUITE_P(Gdn, GdnChunkedPrefill,
        testing::Values(chunked_case_t {kQwen36, {1, 63, 64, 65}, kKv, false},
                chunked_case_t {kQwen36, {129, 200}, kKv, false},
                chunked_case_t {kQwen36, {130}, kVk, false},
                chunked_case_t {kQwen36, {70, 140}, kKv, true},
                chunked_case_t {
                        shape_t {2, 4, 64, 32, 4}, {5, 96, 33}, kKv, false},
                // f16 and f32 sources take FP32 operands in the GEMMs.
                chunked_case_t {kQwen36, {1, 63, 64, 65}, kKv, false, kF16},
                chunked_case_t {kQwen36, {130}, kVk, false, kF16, kBf16},
                chunked_case_t {kQwen36, {70, 140}, kKv, true, kF16, kF16},
                chunked_case_t {shape_t {2, 4, 64, 32, 4}, {5, 96, 33}, kKv,
                        true, kF16},
                chunked_case_t {kQwen36, {129, 200}, kKv, false, kF32},
                chunked_case_t {kQwen36, {130}, kVk, true, kF32, kF16},
                chunked_case_t {shape_t {2, 4, 64, 32, 4}, {5, 96, 33}, kKv,
                        false, kF32, kBf16},
                // Threads that (sequence, value head) pairs would leave
                // idle: value blocks of 64 or 32 columns, for every pair or
                // for those of the last round, every source and state type,
                // kv and vk, with and without the norm.
                chunked_case_t {
                        kQwen36, {130}, kKv, false, kBf16, kF32, 64, "2x32"},
                chunked_case_t {
                        kQwen36, {200}, kVk, true, kBf16, kBf16, 128, "4x32"},
                chunked_case_t {
                        kQwen36, {130}, kVk, false, kF16, kF16, 64, "2x32"},
                chunked_case_t {
                        kQwen36, {70, 140}, kKv, true, kF16, kF32, 128, "2x64"},
                chunked_case_t {
                        kQwen36, {129}, kKv, false, kF32, kBf16, 128, "4x32"},
                chunked_case_t {
                        kQwen36, {65}, kVk, true, kF32, kF32, 64, "2x32"},
                chunked_case_t {shape_t {2, 4, 64, 64, 4}, {96, 33}, kKv, true,
                        kBf16, kF32, 16, "2x8"},
                chunked_case_t {
                        kQwen38, {130}, kKv, false, kBf16, kF32, 32, "2x16"},
                chunked_case_t {
                        kQwen38, {150}, kVk, true, kF16, kBf16, 32, "2x16"},
                chunked_case_t {kQwen36, {100, 100, 100}, kKv, true, kF32, kF32,
                        64, "2x32"},
                // The pairs already fill the threads: whole heads.
                chunked_case_t {kQwen36, {1, 63, 64, 65}, kKv, false, kBf16,
                        kF32, 64, "1"},
                chunked_case_t {
                        kQwen36, {130, 129}, kVk, false, kF16, kF32, 64, "1"}),
        chunked_name);

struct mixed_case_t {
    shape_t s;
    int64_t nd; // decode sequences, leading the batch
    std::vector<int32_t> lens; // prefill lengths
    data_type_t state_dt;
    ng::gdn_state_layout_t layout;
    bool fuse_norm;
    ng::gdn_prefill_algo_t algo;
    data_type_t src = data_type_t::bf16;
    // Threads of the native call (0: the OpenMP default) and the split of
    // its chunked prefill (see chunk_split; null: not checked).
    int32_t threads = 0;
    const char *split = nullptr;
};

std::string mixed_name(const testing::TestParamInfo<mixed_case_t> &info) {
    const mixed_case_t &c = info.param;
    std::ostringstream os;
    os << c.s << "_D" << c.nd << "_P" << c.lens.size() << "_"
       << types_tag(c.src, c.state_dt) << "_"
       << (c.layout == ng::gdn_state_layout_t::kv ? "kv" : "vk")
       << (c.fuse_norm ? "_norm" : "") << "_"
       << ng::prefill_algo_to_string(c.algo);
    if (c.threads > 0) { os << "_T" << c.threads; }
    os << "_" << info.index;
    return os.str();
}

// Decode rows first, then the prefill rows, as vLLM orders a batch.
std::vector<int32_t> mixed_offsets(
        int64_t nd, const std::vector<int32_t> &lens) {
    std::vector<int32_t> all(nd, 1);
    all.insert(all.end(), lens.begin(), lens.end());
    return offsets(all);
}

class GdnMixed : public testing::TestWithParam<mixed_case_t> {};

// One call over decodes and prefills, native against reference: pad slots in
// both parts, decode flags that must not be read, mixed initial state, and
// recurrent or chunked prefills (automatic or forced).
TEST_P(GdnMixed, NativeMatchesReference) {
    GDN_SKIP_UNLESS_NATIVE();
    const mixed_case_t &c = GetParam();
    const std::vector<int32_t> qsl = mixed_offsets(c.nd, c.lens);
    const int64_t np = static_cast<int64_t>(c.lens.size());
    const int64_t n = c.nd + np;
    problem_t ref(c.s, qsl.back(), n + 3, c.state_dt, c.layout, c.fuse_norm, 23,
            c.src);
    ref.idx = distinct_slots(n, n + 3, 6);
    if (c.nd > 2) { ref.idx[1] = -1; }
    if (np > 2) { ref.idx[c.nd + 1] = -1; }
    std::vector<uint8_t> init(n);
    for (int64_t i = 0; i < n; ++i) {
        init[i] = static_cast<uint8_t>(i < c.nd ? i % 2 : (i - c.nd) % 2 == 0);
    }
    problem_t nat = ref;
    nat.p.prefill_algo = c.algo;
    nat.p.num_threads = c.threads;
    if (c.split != nullptr) {
        ASSERT_EQ(chunk_split(nat.p, nat.idx.data(), qsl, c.nd, n, c.threads),
                c.split);
    }
    ASSERT_EQ(ref.mixed(ng::gdn_kernel_t::reference, c.nd, qsl, init),
            status_t::success);
    ASSERT_EQ(nat.mixed(ng::gdn_kernel_t::native, c.nd, qsl, init),
            status_t::success);
    // The chunked form is held to the precision of its operands (see
    // GdnChunkedPrefill); automatic selection may take it once the prefills
    // average 16 tokens (more for f16 and f32 sources).
    const bool may_chunk = c.algo == ng::gdn_prefill_algo_t::chunked
            || (c.algo == ng::gdn_prefill_algo_t::automatic && np > 0
                    && qsl.back() - c.nd >= 16 * np);
    expect_tol(nat, ref,
            may_chunk ? chunked_tol(c.src, c.state_dt, c.fuse_norm)
                      : native_tol(c.src, c.state_dt));
}

INSTANTIATE_TEST_SUITE_P(Gdn, GdnMixed,
        testing::Values(
                // Prefills average 36 tokens: chunked.
                mixed_case_t {kQwen36, 4, {37, 1, 70}, data_type_t::f32,
                        ng::gdn_state_layout_t::kv, false,
                        ng::gdn_prefill_algo_t::automatic},
                // Prefills average 3.5 tokens: recurrent.
                mixed_case_t {kQwen36, 3, {5, 2}, data_type_t::f32,
                        ng::gdn_state_layout_t::kv, false,
                        ng::gdn_prefill_algo_t::automatic},
                mixed_case_t {kQwen36, 8, {130}, data_type_t::bf16,
                        ng::gdn_state_layout_t::vk, false,
                        ng::gdn_prefill_algo_t::automatic},
                mixed_case_t {kQwen36, 2, {9, 20}, data_type_t::f32,
                        ng::gdn_state_layout_t::kv, true,
                        ng::gdn_prefill_algo_t::recurrent},
                mixed_case_t {kQwen36, 5, {70, 0, 64}, data_type_t::f32,
                        ng::gdn_state_layout_t::kv, true,
                        ng::gdn_prefill_algo_t::chunked},
                mixed_case_t {kSmall, 6, {5, 1, 2, 64}, data_type_t::f32,
                        ng::gdn_state_layout_t::kv, false,
                        ng::gdn_prefill_algo_t::automatic},
                mixed_case_t {kSmall, 1, {3}, data_type_t::bf16,
                        ng::gdn_state_layout_t::vk, true,
                        ng::gdn_prefill_algo_t::recurrent},
                // head_v_dim 48 rules out the chunked form.
                mixed_case_t {kNoConv, 3, {3, 17}, data_type_t::f32,
                        ng::gdn_state_layout_t::kv, false,
                        ng::gdn_prefill_algo_t::automatic},
                // f16 and f32 sources: long prefills (chunked) and short
                // ones (recurrent), automatic and forced, every state type.
                mixed_case_t {kQwen36, 4, {37, 1, 70}, kF32, kKv, false,
                        ng::gdn_prefill_algo_t::automatic, kF16},
                mixed_case_t {kQwen36, 3, {5, 2}, kF16, kKv, true,
                        ng::gdn_prefill_algo_t::automatic, kF16},
                mixed_case_t {kQwen36, 8, {130}, kBf16, kVk, false,
                        ng::gdn_prefill_algo_t::automatic, kF16},
                mixed_case_t {kQwen36, 5, {70, 0, 64}, kF16, kKv, true,
                        ng::gdn_prefill_algo_t::chunked, kF16},
                mixed_case_t {kSmall, 6, {5, 1, 2, 64}, kF32, kKv, true,
                        ng::gdn_prefill_algo_t::recurrent, kF16},
                mixed_case_t {kQwen36, 4, {37, 1, 70}, kF32, kKv, true,
                        ng::gdn_prefill_algo_t::automatic, kF32},
                mixed_case_t {kQwen36, 2, {9, 20}, kF16, kVk, false,
                        ng::gdn_prefill_algo_t::recurrent, kF32},
                mixed_case_t {kSmall, 1, {3, 96}, kBf16, kKv, false,
                        ng::gdn_prefill_algo_t::chunked, kF32},
                // Chunked prefills in value blocks next to decodes, with a
                // padded prefill and every source type.
                mixed_case_t {kQwen36, 4, {130}, kF32, kKv, false,
                        ng::gdn_prefill_algo_t::chunked, kBf16, 64, "2x32"},
                mixed_case_t {kQwen36, 3, {200, 5, 70}, kBf16, kVk, true,
                        ng::gdn_prefill_algo_t::chunked, kF16, 128, "2x96"},
                mixed_case_t {kQwen36, 2, {150}, kF16, kKv, true,
                        ng::gdn_prefill_algo_t::automatic, kF32, 128, "4x32"},
                mixed_case_t {kQwen38, 3, {140}, kF32, kVk, false,
                        ng::gdn_prefill_algo_t::automatic, kBf16, 32, "2x16"}),
        mixed_name);

// The kernels gdn_direct runs: the native ones when the CPU has them, and
// the reference.
std::vector<ng::gdn_kernel_t> kernels_to_check() {
    std::vector<ng::gdn_kernel_t> ks {ng::gdn_kernel_t::reference};
    if (has_native_isa()) { ks.push_back(ng::gdn_kernel_t::native); }
    return ks;
}

void expect_identical(problem_t &x, problem_t &y, const char *what) {
    SCOPED_TRACE(what);
    expect_close(x.out, y.out, 0.0f, 0.0f, "output");
    expect_close(x.ssm_state, y.ssm_state, 0.0f, 0.0f, "ssm_state");
    expect_close(x.conv_state, y.conv_state, 0.0f, 0.0f, "conv_state");
}

// A mixed call gives exactly what a decode-only call on its decodes followed
// by a prefill-only call on its prefills (rows rebased) gives.
TEST(GdnMixedContract, EqualsSeparateDecodeAndPrefillCalls) {
    const int64_t nd = 5;
    const std::vector<int32_t> lens {40, 1, 90};
    const std::vector<int32_t> qsl = mixed_offsets(nd, lens);
    const int64_t np = static_cast<int64_t>(lens.size()), n = nd + np;
    const std::vector<uint8_t> init {1, 1, 1, 1, 1, 1, 0, 1};
    for (ng::gdn_kernel_t k : kernels_to_check()) {
        problem_t x(kQwen36, qsl.back(), n + 2, data_type_t::f32,
                ng::gdn_state_layout_t::kv, true, 29);
        x.idx = distinct_slots(n, n + 2, 8);
        x.idx[2] = -1;
        problem_t y = x;
        ASSERT_EQ(x.mixed(k, nd, qsl, init), status_t::success);

        ng::gdn_params qd = y.p;
        qd.batch = nd;
        qd.num_decodes = nd;
        qd.num_tokens = nd;
        ASSERT_EQ(y.call(qd, k), status_t::success);

        const int64_t r0 = qsl[nd];
        std::vector<int32_t> local(np + 1);
        for (int64_t i = 0; i <= np; ++i) {
            local[i] = qsl[nd + i] - static_cast<int32_t>(r0);
        }
        ng::gdn_params qp = y.p;
        qp.kernel = k;
        qp.batch = np;
        qp.num_decodes = 0;
        qp.num_tokens = local.back();
        qp.cu_seqlens = local.data();
        qp.has_initial_state = reinterpret_cast<const bool *>(init.data()) + nd;
        qp.norm_weight = y.norm_w.data();
        qp.z = static_cast<const uint16_t *>(y.z.data()) + r0 * qp.z_stride_t;
        const auto row = [r0](buf_t &buf, int64_t stride, int64_t col = 0) {
            return static_cast<uint16_t *>(buf.data()) + r0 * stride + col;
        };
        ASSERT_EQ(
                ng::gdn_direct(row(y.qkv, qp.qkv_stride), row(y.a, qp.a_stride),
                        row(y.b, qp.b_stride, y.s.HV), y.a_log.data(),
                        y.dt_bias.data(), y.conv_w.data(), y.conv_b.data(),
                        y.conv_state.data(), y.ssm_state.data(),
                        y.idx.data() + nd, row(y.out, qp.out_stride_t), qp),
                status_t::success);
        expect_identical(x, y, ng::kernel_to_string(k));
    }
}

// The has_initial_state flags of decodes are not read (a decode always
// continues from its slot), a null has_initial_state starts every prefill
// from zero, and entries past the batch in cu_seqlens / has_initial_state
// (vLLM pads both) are not read.
TEST(GdnMixedContract, DecodeFlagsAndPaddingAreNotRead) {
    const int64_t nd = 3;
    const std::vector<int32_t> qsl = mixed_offsets(nd, {6, 2});
    for (ng::gdn_kernel_t k : kernels_to_check()) {
        SCOPED_TRACE(ng::kernel_to_string(k));
        problem_t base(kSmall, qsl.back(), 7, data_type_t::f32,
                ng::gdn_state_layout_t::kv, false, 37);
        base.idx = distinct_slots(nd + 2, 7, 10);
        problem_t x = base, y = base, z = base, w = base;
        ASSERT_EQ(x.mixed(k, nd, qsl, {1, 1, 1, 0, 0}), status_t::success);
        ASSERT_EQ(y.mixed(k, nd, qsl, {0, 1, 0, 0, 0}), status_t::success);
        ASSERT_EQ(z.mixed(k, nd, qsl, {}, true), status_t::success);
        std::vector<int32_t> padded = qsl;
        padded.insert(padded.end(), {qsl.back(), qsl.back()});
        ASSERT_EQ(w.mixed(k, nd, padded, {1, 1, 1, 0, 0, 1, 1}),
                status_t::success);
        expect_identical(y, x, "decode flags");
        expect_identical(z, x, "null has_initial_state");
        expect_identical(w, x, "padded metadata");
    }
}

// Without cu_seqlens every sequence is one token, and the single-token
// prefills after the decodes run the prefill kernel.
TEST(GdnMixedContract, NullCuSeqlensIsOneTokenPerSequence) {
    const std::vector<uint8_t> init {1, 0, 1, 0, 1};
    for (ng::gdn_kernel_t k : kernels_to_check()) {
        SCOPED_TRACE(ng::kernel_to_string(k));
        problem_t x(kSmall, 5, 7, data_type_t::f32, ng::gdn_state_layout_t::kv,
                false, 31);
        x.idx = distinct_slots(5, 7, 9);
        problem_t y = x;
        ASSERT_EQ(x.mixed(k, 2, {}, init), status_t::success);
        ASSERT_EQ(y.mixed(k, 2, offsets({1, 1, 1, 1, 1}), init),
                status_t::success);
        expect_identical(x, y, "null vs unit cu_seqlens");
    }
}

// The split of the chunked prefill (Qwen3.6 and Qwen3.8 heads, bf16): whole
// heads while the (sequence, value head) pairs keep the threads busy; when
// threads would idle, value blocks for every pair or for the pairs of the
// last round, at most head_v_dim / 32 blocks; pad slots and empty sequences
// do not count.
TEST(GdnChunkedValueBlocks, Selection) {
    ng::gdn_params p;
    p.num_k_heads = 16;
    p.num_v_heads = 32;
    p.head_k_dim = 128;
    p.head_v_dim = 128;
    p.src_dt = data_type_t::bf16;
    const auto split = [&p](const std::vector<int32_t> &lens, int32_t threads,
                               std::vector<int32_t> slots = {}) {
        if (slots.empty()) {
            slots.resize(lens.size());
            std::iota(slots.begin(), slots.end(), 0);
        }
        return chunk_split(p, slots.data(), offsets(lens), 0,
                static_cast<int64_t>(lens.size()), threads);
    };
    // One 4096-token prompt: 32 pairs.
    EXPECT_EQ(split({4096}, 1), "1");
    EXPECT_EQ(split({4096}, 32), "1");
    EXPECT_EQ(split({4096}, 48), "1");
    EXPECT_EQ(split({4096}, 64), "2x32");
    EXPECT_EQ(split({4096}, 96), "2x32");
    EXPECT_EQ(split({4096}, 128), "4x32");
    // Prompts that fill the threads, or a last round of half of them.
    EXPECT_EQ(split({2048, 2048}, 64), "1");
    EXPECT_EQ(split({512, 512, 512, 512, 512, 512, 512, 512}, 64), "1");
    EXPECT_EQ(split({1204, 1204, 1204}, 64), "2x32");
    // A long prompt next to a short one; a padded and an empty one.
    EXPECT_EQ(split({4096, 256}, 64), "2x64");
    EXPECT_EQ(split({4096, 300, 0}, 64, {0, -1, 2}), "2x96");
    // 48 pairs: on 32 threads the 16 of the second round are split.
    p.num_v_heads = 48;
    EXPECT_EQ(split({4096}, 32), "2x16");
    EXPECT_EQ(split({4096}, 48), "1");
    EXPECT_EQ(split({4096}, 64), "1");
    p.num_v_heads = 32;
    p.head_v_dim = 64;
    EXPECT_EQ(split({4096}, 128), "2x32");
    p.head_v_dim = 96;
    EXPECT_EQ(split({4096}, 128), "1");
    p.head_v_dim = 32;
    EXPECT_EQ(split({4096}, 128), "1");
}

// A unit computes its block of value columns exactly as a whole-head unit
// computes them, so the split changes no bit of the outputs or the states:
// Qwen3.6 heads whole on 1 thread, in 2 blocks on 64 and in 4 on 128, and
// Qwen3.8 heads (48 pairs) with the last round's 16 pairs in 2 blocks on 32.
TEST(GdnChunkedValueBlocks, BitExactAcrossSplits) {
    GDN_SKIP_UNLESS_NATIVE();
    struct cfg_t {
        shape_t s;
        data_type_t src, state_dt;
        ng::gdn_state_layout_t layout;
        bool norm;
    };
    const cfg_t cfgs[] = {{kQwen36, kBf16, kF32, kKv, false},
            {kQwen36, kBf16, kBf16, kVk, true},
            {kQwen36, kF16, kF32, kKv, true}, {kQwen36, kF16, kF16, kVk, false},
            {kQwen36, kF32, kF32, kVk, true},
            {kQwen36, kF32, kBf16, kKv, false},
            {kQwen38, kBf16, kF32, kKv, false},
            {kQwen38, kF16, kF32, kVk, true}};
    // A 130-token prompt (its last chunk partial), a padded prompt and an
    // empty one.
    const std::vector<int32_t> qsl = offsets({130, 5, 0});
    const std::vector<uint8_t> init {1, 1, 0};
    for (const cfg_t &c : cfgs) {
        SCOPED_TRACE(testing::Message()
                << c.s << " " << types_tag(c.src, c.state_dt)
                << (c.layout == kKv ? " kv" : " vk")
                << (c.norm ? " norm" : ""));
        problem_t base(
                c.s, qsl.back(), 5, c.state_dt, c.layout, c.norm, 43, c.src);
        base.idx = distinct_slots(3, 5, 13);
        base.idx[1] = -1;
        base.p.prefill_algo = ng::gdn_prefill_algo_t::chunked;
        problem_t whole = base;
        whole.p.num_threads = 1;
        ASSERT_EQ(whole.prefill(ng::gdn_kernel_t::native, qsl, init),
                status_t::success);
        const int64_t pairs = 3 * c.s.HV;
        const std::vector<std::pair<int32_t, std::string>> runs = c.s.HV == 48
                ? std::vector<std::pair<int32_t, std::string>> {{32, "2x112"}}
                : std::vector<std::pair<int32_t, std::string>> {
                          {64, "2x" + std::to_string(pairs)},
                          {128, "4x" + std::to_string(pairs)}};
        for (const auto &run : runs) {
            problem_t x = base;
            x.p.num_threads = run.first;
            ASSERT_EQ(chunk_split(x.p, x.idx.data(), qsl, 0, 3, run.first),
                    run.second);
            ASSERT_EQ(x.prefill(ng::gdn_kernel_t::native, qsl, init),
                    status_t::success);
            expect_identical(x, whole, run.second.c_str());
        }
    }
}

TEST(GdnVariants, ConvWidths) {
    GDN_SKIP_UNLESS_NATIVE();
    for (int64_t w : {1, 2, 3, 8}) {
        SCOPED_TRACE(testing::Message() << "W " << w);
        check_variant({2, 4, 32, 32, w}, data_type_t::bf16, data_type_t::f32,
                ng::gdn_state_layout_t::kv, false, nullptr);
    }
}

// A forced native request on a shape the native kernels do not cover must
// fall back to the reference and give identical results.
TEST(GdnVariants, UnsupportedShapeFallsBackToReference) {
    if (kernel_env_set()) { GTEST_SKIP() << "ZENDNNL_GDN_KERNEL is set"; }
    for (const shape_t &s : {shape_t {2, 2, 40, 32, 4},
                 shape_t {1, 1, 32, 272, 4}, shape_t {1, 2, 32, 32, 9}}) {
        SCOPED_TRACE(testing::Message() << s);
        problem_t ref(s, 3, 4, data_type_t::f32, ng::gdn_state_layout_t::kv,
                false, 5);
        ref.idx = distinct_slots(3, 4, 1);
        problem_t nat = ref;
        ASSERT_EQ(ref.decode(ng::gdn_kernel_t::reference), status_t::success);
        ASSERT_EQ(nat.decode(ng::gdn_kernel_t::native), status_t::success);
        expect_same(nat, ref, 0.0f, "decode");
    }
}

TEST(GdnPrefillContract, NullHasInitialStateIsZeroStart) {
    for (ng::gdn_kernel_t k :
            {ng::gdn_kernel_t::reference, ng::gdn_kernel_t::native}) {
        if (k == ng::gdn_kernel_t::native && !has_native_isa()) { continue; }
        const std::vector<int32_t> qsl = offsets({4, 7});
        problem_t x(kSmall, qsl.back(), 3, data_type_t::f32,
                ng::gdn_state_layout_t::kv, false, 9);
        x.idx = {2, 0};
        problem_t y = x;
        ASSERT_EQ(x.prefill(k, qsl, {0, 0}), status_t::success);
        ASSERT_EQ(y.prefill(k, qsl, {}, true), status_t::success);
        expect_same(y, x, 0.0f, ng::kernel_to_string(k));
    }
}

// Pad slots and empty sequences leave their output rows and pool slots alone.
TEST(GdnPrefillContract, PadSlotAndEmptySequenceAreSkipped) {
    GDN_SKIP_UNLESS_NATIVE();
    const std::vector<int32_t> qsl = offsets({5, 0, 7, 3});
    problem_t ref(kQwen36, qsl.back(), 6, data_type_t::f32,
            ng::gdn_state_layout_t::kv, false, 13);
    ref.idx = {4, 1, 0, -1};
    const problem_t before = ref;
    problem_t nat = ref;
    ASSERT_EQ(ref.prefill(ng::gdn_kernel_t::reference, qsl, {1, 1, 0, 1}),
            status_t::success);
    ASSERT_EQ(nat.prefill(ng::gdn_kernel_t::native, qsl, {1, 1, 0, 1}),
            status_t::success);
    expect_same(nat, ref, 2e-2f, "native vs reference");
    // Slot 1 (empty sequence) and the pad sequence's rows are untouched.
    const int64_t hs = kQwen36.HV * kQwen36.K * kQwen36.V;
    for (int64_t i = hs; i < 2 * hs; ++i) {
        ASSERT_EQ(nat.ssm_state.get(i), before.ssm_state.get(i));
    }
    for (int64_t t = qsl[3]; t < qsl[4]; ++t) {
        ASSERT_TRUE(std::isnan(nat.out.get(t * nat.p.out_stride_t)));
    }
}

// update_state = false: the prefills give the outputs of update_state = true
// and leave both pools as they were.
TEST(GdnPrefillContract, UpdateStateFalseLeavesPoolsUnchanged) {
    if (kernel_env_set() || prefill_env_set()) {
        GTEST_SKIP() << "ZENDNNL_GDN_KERNEL / ZENDNNL_GDN_PREFILL is set";
    }
    const std::vector<int32_t> qsl = offsets({70, 9, 33});
    for (ng::gdn_kernel_t k : kernels_to_check()) {
        for (ng::gdn_prefill_algo_t algo : {ng::gdn_prefill_algo_t::recurrent,
                     ng::gdn_prefill_algo_t::chunked}) {
            SCOPED_TRACE(std::string(ng::kernel_to_string(k)) + " "
                    + ng::prefill_algo_to_string(algo));
            problem_t x(kSmall, qsl.back(), 5, kF32, kKv, true, 31);
            x.idx = {3, 0, 4};
            x.p.prefill_algo = algo;
            problem_t y = x;
            const problem_t before = x;
            y.p.update_state = false;
            ASSERT_EQ(x.prefill(k, qsl, {1, 0, 1}), status_t::success);
            ASSERT_EQ(y.prefill(k, qsl, {1, 0, 1}), status_t::success);
            expect_close(y.out, x.out, 0.0f, 0.0f, "output");
            expect_close(
                    y.ssm_state, before.ssm_state, 0.0f, 0.0f, "ssm_state");
            expect_close(
                    y.conv_state, before.conv_state, 0.0f, 0.0f, "conv_state");
        }
    }
}

// With update_state = false several prefills may continue from one slot:
// each gives what it gives alone from that slot.
TEST(GdnPrefillContract, UpdateStateFalseSequencesShareASlot) {
    if (kernel_env_set() || prefill_env_set()) {
        GTEST_SKIP() << "ZENDNNL_GDN_KERNEL / ZENDNNL_GDN_PREFILL is set";
    }
    const std::vector<int32_t> qsl = offsets({40, 7, 23});
    for (ng::gdn_kernel_t k : kernels_to_check()) {
        for (ng::gdn_prefill_algo_t algo : {ng::gdn_prefill_algo_t::recurrent,
                     ng::gdn_prefill_algo_t::chunked}) {
            SCOPED_TRACE(std::string(ng::kernel_to_string(k)) + " "
                    + ng::prefill_algo_to_string(algo));
            problem_t shared(kSmall, qsl.back(), 2, kF32, kKv, true, 41);
            shared.p.prefill_algo = algo;
            const problem_t base = shared;
            shared.idx = {1, 1, 1};
            shared.p.update_state = false;
            ASSERT_EQ(shared.prefill(k, qsl, {1, 1, 1}), status_t::success);
            expect_close(
                    shared.ssm_state, base.ssm_state, 0.0f, 0.0f, "ssm_state");
            const int64_t row = shared.p.out_stride_t;
            for (size_t i = 0; i + 1 < qsl.size(); ++i) {
                // Sequence i alone: the others are pad slots.
                problem_t one = base;
                one.idx = {-1, -1, -1};
                one.idx[i] = 1;
                ASSERT_EQ(one.prefill(k, qsl, {1, 1, 1}), status_t::success);
                for (int64_t e = qsl[i] * row; e < qsl[i + 1] * row; ++e) {
                    ASSERT_EQ(shared.out.get(e), one.out.get(e))
                            << "sequence " << i << ", element " << e;
                }
            }
        }
    }
}

// Pools nothing reads or writes may be null. An empty prefill reads nothing,
// whatever its has_initial_state flag.
TEST(GdnPrefillContract, UpdateStateFalseWithoutInitialStateNeedsNoPools) {
    const std::vector<int32_t> qsl = offsets({0, 30, 12});
    const std::vector<uint8_t> init {1, 0, 0};
    for (ng::gdn_kernel_t k : kernels_to_check()) {
        for (bool flags : {false, true}) {
            SCOPED_TRACE(std::string(ng::kernel_to_string(k))
                    + (flags ? " flags" : " null flags"));
            problem_t x(kSmall, qsl.back(), 3, kF32, kKv, true, 51);
            x.idx = {2, 0, 1};
            problem_t y = x;
            ASSERT_EQ(x.prefill(k, qsl, init, !flags), status_t::success);
            ng::gdn_params q = y.p;
            q.batch = 3;
            q.num_tokens = qsl.back();
            q.cu_seqlens = qsl.data();
            q.has_initial_state = flags
                    ? reinterpret_cast<const bool *>(init.data())
                    : nullptr;
            q.update_state = false;
            q.kernel = k;
            q.norm_weight = y.norm_w.data();
            q.z = y.z.data();
            ASSERT_EQ(ng::gdn_direct(y.qkv.data(), y.a.data(), y.b_ptr(),
                              y.a_log.data(), y.dt_bias.data(), y.conv_w.data(),
                              y.conv_b.data(), nullptr, nullptr, y.idx.data(),
                              y.out.data(), q),
                    status_t::success);
            expect_close(y.out, x.out, 0.0f, 0.0f, "output");
        }
    }
}

// The inputs of FLA's / transformers' chunk_gated_delta_rule: q, k and v as
// separate tensors with padded head and token strides, and g / beta computed
// from a / b outside the kernel, give what the packed rows with the gate
// projections give.
struct rule_inputs_t {
    buf_t q, k, v, g, beta;
    int64_t qs_h, qs_t, ks_h, ks_t, vs_h, vs_t;
};

rule_inputs_t rule_inputs(
        const problem_t &x, data_type_t g_dt, data_type_t beta_dt) {
    const shape_t &s = x.s;
    rule_inputs_t r;
    r.qs_h = s.K + 16;
    r.qs_t = s.H * r.qs_h + 32;
    r.ks_h = s.K;
    r.ks_t = s.H * s.K + 16;
    r.vs_h = s.V + 32;
    r.vs_t = s.HV * r.vs_h;
    r.q = buf_t(x.qkv.dt, x.tokens * r.qs_t);
    r.k = buf_t(x.qkv.dt, x.tokens * r.ks_t);
    r.v = buf_t(x.qkv.dt, x.tokens * r.vs_t);
    r.g = buf_t(g_dt, x.tokens * s.HV);
    r.beta = buf_t(beta_dt, x.tokens * s.HV);
    const int64_t qk = s.H * s.K;
    // Where b_ptr() puts b in each ab row, in elements of b's type.
    const int64_t b0 = (static_cast<const uint16_t *>(x.b_ptr())
                               - static_cast<const uint16_t *>(x.b.data()))
            / (x.b.dt == data_type_t::f32 ? 2 : 1);
    for (int64_t t = 0; t < x.tokens; ++t) {
        const int64_t row = t * x.p.qkv_stride;
        for (int64_t h = 0; h < s.H; ++h) {
            for (int64_t i = 0; i < s.K; ++i) {
                r.q.set(t * r.qs_t + h * r.qs_h + i,
                        x.qkv.get(row + h * s.K + i));
                r.k.set(t * r.ks_t + h * r.ks_h + i,
                        x.qkv.get(row + qk + h * s.K + i));
            }
        }
        for (int64_t hv = 0; hv < s.HV; ++hv) {
            for (int64_t j = 0; j < s.V; ++j) {
                r.v.set(t * r.vs_t + hv * r.vs_h + j,
                        x.qkv.get(row + 2 * qk + hv * s.V + j));
            }
            const float a = x.a.get(t * x.p.a_stride + hv);
            const float b = x.b.get(t * x.p.b_stride + b0 + hv);
            const float sp = a + x.dt_bias.get(hv);
            const float g = -std::exp(x.a_log.get(hv))
                    * (sp > 20.0f ? sp : std::log1p(std::exp(sp)));
            r.g.set(t * s.HV + hv, g);
            r.beta.set(t * s.HV + hv, 1.0f / (1.0f + std::exp(-b)));
        }
    }
    return r;
}

size_t words(const buf_t &buf) {
    return buf.dt == data_type_t::f32 ? 2 : 1;
}

// gdn_direct on y's token rows from r0 on, as the batch qsl describes
// (sequence i on slot y.idx[i], the first nd sequences decodes); with r, on
// its separate q / k / v and precomputed gates instead of y's packed rows and
// gate projections.
status_t call_rows(problem_t &y, const rule_inputs_t *r, int64_t r0,
        const std::vector<int32_t> &qsl, const std::vector<uint8_t> &init,
        ng::gdn_kernel_t kernel, int64_t nd = 0) {
    ng::gdn_params q = y.p;
    q.kernel = kernel;
    q.batch = static_cast<int64_t>(y.idx.size());
    q.num_decodes = nd;
    q.num_tokens = qsl.back();
    q.cu_seqlens = qsl.data();
    q.has_initial_state = reinterpret_cast<const bool *>(init.data());
    const auto row = [r0](const buf_t &buf, int64_t stride) {
        return static_cast<const uint16_t *>(buf.data())
                + r0 * stride * static_cast<int64_t>(words(buf));
    };
    if (q.norm_weight) {
        q.norm_weight = y.norm_w.data();
        q.z = row(y.z, y.s.HV * y.s.V);
    }
    void *out = const_cast<uint16_t *>(row(y.out, q.out_stride_t));
    if (r == nullptr) {
        const uint16_t *b = static_cast<const uint16_t *>(y.b_ptr())
                + r0 * q.b_stride * static_cast<int64_t>(words(y.b));
        return ng::gdn_direct(row(y.qkv, q.qkv_stride), row(y.a, q.a_stride), b,
                y.a_log.data(), y.dt_bias.data(), y.conv_w.data(),
                y.conv_b.data(), y.conv_state.data(), y.ssm_state.data(),
                y.idx.data(), out, q);
    }
    q.qkv_stride = r->qs_t;
    q.q_stride_h = r->qs_h;
    q.k = row(r->k, r->ks_t);
    q.k_stride_t = r->ks_t;
    q.k_stride_h = r->ks_h;
    q.v = row(r->v, r->vs_t);
    q.v_stride_t = r->vs_t;
    q.v_stride_h = r->vs_h;
    q.gate_input = ng::gdn_gate_input_t::precomputed;
    q.a_dt = r->g.dt;
    q.b_dt = r->beta.dt;
    q.a_stride = y.s.HV;
    q.b_stride = y.s.HV;
    return ng::gdn_direct(row(r->q, r->qs_t), row(r->g, y.s.HV),
            row(r->beta, y.s.HV), nullptr, nullptr, y.conv_w.data(),
            y.conv_b.data(), y.conv_state.data(), y.ssm_state.data(),
            y.idx.data(), out, q);
}

status_t rule_call(problem_t &y, rule_inputs_t &r, ng::gdn_kernel_t kernel,
        const std::vector<int32_t> &qsl, const std::vector<uint8_t> &init) {
    return call_rows(y, &r, 0, qsl, init, kernel);
}

TEST(GdnRuleInputs, SeparateQkvAndPrecomputedGatesMatchPackedRows) {
    if (kernel_env_set() || prefill_env_set()) {
        GTEST_SKIP() << "ZENDNNL_GDN_KERNEL / ZENDNNL_GDN_PREFILL is set";
    }
    // Already convolved inputs, GQA 2, head dims the chunked form takes.
    const shape_t s {2, 4, 64, 32, 0};
    const std::vector<int32_t> qsl = offsets({70, 9, 33, 1});
    const std::vector<uint8_t> init {1, 0, 1, 1};
    struct case_t {
        data_type_t src, g_dt, beta_dt;
        bool norm;
        float tol;
    };
    for (const case_t &c : {case_t {kF32, kF32, kF32, false, 1e-4f},
                 case_t {kF32, kF32, kF32, true, 1e-4f},
                 case_t {data_type_t::bf16, kF32, data_type_t::bf16, false,
                         3e-2f},
                 case_t {data_type_t::f16, kF32, kF32, true, 6e-3f}}) {
        for (ng::gdn_kernel_t k : kernels_to_check()) {
            for (ng::gdn_prefill_algo_t algo :
                    {ng::gdn_prefill_algo_t::recurrent,
                            ng::gdn_prefill_algo_t::chunked}) {
                SCOPED_TRACE(std::string(ng::kernel_to_string(k)) + " "
                        + ng::prefill_algo_to_string(algo) + " " + dt_tag(c.src)
                        + (c.norm ? " norm" : ""));
                problem_t x(s, qsl.back(), 6, kF32, kKv, c.norm, 61, c.src);
                x.idx = {4, 0, 5, 2};
                x.p.prefill_algo = algo;
                problem_t y = x;
                rule_inputs_t r = rule_inputs(y, c.g_dt, c.beta_dt);
                ASSERT_EQ(x.prefill(k, qsl, init), status_t::success);
                ASSERT_EQ(rule_call(y, r, k, qsl, init), status_t::success);
                expect_close(y.out, x.out, c.tol, c.tol, "output");
                expect_close(y.ssm_state, x.ssm_state, std::max(c.tol, 2e-3f),
                        std::max(c.tol, 1e-2f), "ssm_state");
            }
        }
    }
}

// Precomputed log decays of either sign (g > 0 grows the state, which the
// layer's own gates never do): both forms of the native prefill match the
// reference kernel.
TEST(GdnRuleInputs, PrecomputedGatesOfEitherSign) {
    if (kernel_env_set() || prefill_env_set()) {
        GTEST_SKIP() << "ZENDNNL_GDN_KERNEL / ZENDNNL_GDN_PREFILL is set";
    }
    if (!has_native_isa()) {
        GTEST_SKIP() << "the native GDN kernels need AVX512-BF16";
    }
    const shape_t s {2, 4, 64, 32, 0};
    const std::vector<int32_t> qsl = offsets({70, 9, 33, 1});
    const std::vector<uint8_t> init {1, 0, 1, 1};
    for (data_type_t src : {kF32, data_type_t::bf16}) {
        problem_t base(s, qsl.back(), 6, kF32, kKv, false, 67, src);
        base.idx = {4, 0, 5, 2};
        rule_inputs_t r = rule_inputs(base, kF32, src);
        // Mostly growing: e^g up to 1.1, about e^1.7 over the longest prefill.
        std::mt19937 rng(11);
        r.g.fill_uniform(rng, -0.05f, 0.1f);
        problem_t ref = base;
        ASSERT_EQ(rule_call(ref, r, ng::gdn_kernel_t::reference, qsl, init),
                status_t::success);
        const float tol = src == kF32 ? 1e-4f : 3e-2f;
        for (ng::gdn_prefill_algo_t algo : {ng::gdn_prefill_algo_t::recurrent,
                     ng::gdn_prefill_algo_t::chunked}) {
            SCOPED_TRACE(std::string(ng::prefill_algo_to_string(algo)) + " "
                    + dt_tag(src));
            problem_t y = base;
            y.p.prefill_algo = algo;
            ASSERT_EQ(rule_call(y, r, ng::gdn_kernel_t::native, qsl, init),
                    status_t::success);
            expect_close(y.out, ref.out, tol, tol, "output");
            expect_close(y.ssm_state, ref.ssm_state, std::max(tol, 2e-3f),
                    std::max(tol, 1e-2f), "ssm_state");
        }
    }
}

// f16 sources: values past f16's range give what the recurrent form gives,
// so the chunk's products make no infinity. Decays that grow to e^16 inside
// a chunk (g = 0.5 over 32 tokens, unit q and k, zero v and state), and an
// FP32 state of 65536 that decays (zero q, k and v).
TEST(GdnRuleInputs, F16ChunkedKeepsFp32Range) {
    if (kernel_env_set() || prefill_env_set()) {
        GTEST_SKIP() << "ZENDNNL_GDN_KERNEL / ZENDNNL_GDN_PREFILL is set";
    }
    if (!has_native_isa()) {
        GTEST_SKIP() << "the native GDN kernels need AVX512-BF16";
    }
    const shape_t s {1, 1, 32, 32, 0};
    const std::vector<int32_t> qsl = offsets({32});
    struct case_t {
        float g, beta, qk, s00;
    };
    for (const case_t &c : {case_t {0.5f, 1.0f, 1.0f, 0.0f},
                 case_t {-std::log(2.0f), 0.5f, 0.0f, 65536.0f}}) {
        SCOPED_TRACE(testing::Message() << "g " << c.g << ", S00 " << c.s00);
        problem_t base(s, qsl.back(), 1, kF32, kKv, false, 71, kF16);
        base.idx = {0};
        rule_inputs_t r = rule_inputs(base, kF32, kF32);
        for (int64_t t = 0; t < qsl.back(); ++t) {
            for (int64_t i = 0; i < s.K; ++i) {
                r.q.set(t * r.qs_t + i, i == 0 ? c.qk : 0.0f);
                r.k.set(t * r.ks_t + i, i == 0 ? c.qk : 0.0f);
            }
            for (int64_t j = 0; j < s.V; ++j) {
                r.v.set(t * r.vs_t + j, 0.0f);
            }
            r.g.set(t, c.g);
            r.beta.set(t, c.beta);
        }
        for (size_t i = 0; i < base.ssm_state.size(); ++i) {
            base.ssm_state.set(i, i == 0 ? c.s00 : 0.0f);
        }
        problem_t ref = base;
        ASSERT_EQ(rule_call(ref, r, ng::gdn_kernel_t::reference, qsl, {1}),
                status_t::success);
        for (ng::gdn_prefill_algo_t algo : {ng::gdn_prefill_algo_t::recurrent,
                     ng::gdn_prefill_algo_t::chunked}) {
            SCOPED_TRACE(ng::prefill_algo_to_string(algo));
            problem_t y = base;
            y.p.prefill_algo = algo;
            ASSERT_EQ(rule_call(y, r, ng::gdn_kernel_t::native, qsl, {1}),
                    status_t::success);
            expect_close(y.out, ref.out, 0.0f, 0.0f, "output");
            expect_close(y.ssm_state, ref.ssm_state, 0.0f, 1e-4f, "ssm_state");
        }
    }
}

// The chunk_gated_delta_rule inputs split into value blocks as packed rows do:
// separate strided q / k / v and given g / beta give the bits of whole heads,
// with q / k as 16 key heads and repeated to the 32 value heads.
TEST(GdnChunkedValueBlocks, RuleInputsBitExactAcrossSplits) {
    GDN_SKIP_UNLESS_NATIVE();
    const std::vector<int32_t> qsl = offsets({130, 5, 0});
    const std::vector<uint8_t> init {1, 1, 0};
    for (const shape_t &s :
            {shape_t {16, 32, 128, 128, 0}, shape_t {32, 32, 128, 128, 0}}) {
        for (data_type_t src : {kBf16, kF16, kF32}) {
            SCOPED_TRACE(testing::Message() << s << " " << dt_tag(src));
            problem_t base(s, qsl.back(), 5, kF32, kKv, false, 47, src);
            base.idx = distinct_slots(3, 5, 17);
            base.idx[1] = -1;
            base.p.prefill_algo = ng::gdn_prefill_algo_t::chunked;
            rule_inputs_t r = rule_inputs(base, kF32, kF32);
            problem_t whole = base;
            whole.p.num_threads = 1;
            ASSERT_EQ(rule_call(whole, r, ng::gdn_kernel_t::native, qsl, init),
                    status_t::success);
            for (const auto &run :
                    std::vector<std::pair<int32_t, std::string>> {
                            {64, "2x96"}, {128, "4x96"}}) {
                problem_t x = base;
                x.p.num_threads = run.first;
                ASSERT_EQ(chunk_split(x.p, x.idx.data(), qsl, 0, 3, run.first),
                        run.second);
                ASSERT_EQ(rule_call(x, r, ng::gdn_kernel_t::native, qsl, init),
                        status_t::success);
                expect_identical(x, whole, run.second.c_str());
            }
        }
    }
}

// update_state = false with two prompts reading one slot (and an empty one):
// the value blocks read the shared state, write nothing back and give the
// bits of whole heads.
TEST(GdnChunkedValueBlocks, UpdateStateFalseBitExactAcrossSplits) {
    GDN_SKIP_UNLESS_NATIVE();
    const std::vector<int32_t> qsl = offsets({130, 130, 0});
    const std::vector<uint8_t> init {1, 1, 0};
    for (data_type_t src : {kBf16, kF32}) {
        SCOPED_TRACE(dt_tag(src));
        problem_t base(kQwen36, qsl.back(), 2, kF32, kKv, true, 53, src);
        base.idx = {1, 1, 1};
        base.p.update_state = false;
        base.p.prefill_algo = ng::gdn_prefill_algo_t::chunked;
        const problem_t pools = base;
        problem_t whole = base;
        whole.p.num_threads = 1;
        ASSERT_EQ(whole.prefill(ng::gdn_kernel_t::native, qsl, init),
                status_t::success);
        expect_close(whole.ssm_state, pools.ssm_state, 0.0f, 0.0f, "ssm_state");
        expect_close(
                whole.conv_state, pools.conv_state, 0.0f, 0.0f, "conv_state");
        for (const auto &run : std::vector<std::pair<int32_t, std::string>> {
                     {128, "2x96"}, {256, "4x96"}}) {
            problem_t x = base;
            x.p.num_threads = run.first;
            ASSERT_EQ(chunk_split(x.p, x.idx.data(), qsl, 0, 3, run.first),
                    run.second);
            ASSERT_EQ(x.prefill(ng::gdn_kernel_t::native, qsl, init),
                    status_t::success);
            expect_identical(x, whole, run.second.c_str());
        }
    }
}

// Split passes: a prefill continued from the state an earlier call left in
// its slot (has_initial_state), as vLLM's prefix caching and decision models
// answering prompts from a shared context run it.

const shape_t kQwen35 {16, 16, 128, 128, 4};

// Copies x's token rows to y: row from[j] of x's per-token tensors (packed
// rows, gate projections, z and output) becomes row j of y's.
void gather_rows(
        const problem_t &x, problem_t &y, const std::vector<int64_t> &from) {
    const auto copy = [&from](const buf_t &src, buf_t &dst, int64_t stride) {
        const size_t n = static_cast<size_t>(stride) * words(src);
        for (size_t j = 0; j < from.size(); ++j) {
            std::memcpy(dst.raw.data() + j * n,
                    src.raw.data() + static_cast<size_t>(from[j]) * n,
                    n * sizeof(uint16_t));
        }
    };
    copy(x.qkv, y.qkv, x.p.qkv_stride);
    copy(x.a, y.a, x.p.a_stride);
    copy(x.b, y.b, x.p.b_stride);
    copy(x.z, y.z, x.s.HV * x.s.V);
    copy(x.out, y.out, x.p.out_stride_t);
}

// A batch run as two calls: every sequence's first cut[i] tokens, then every
// rest, continuing from the state the first call left. Row j of the split
// rows holds row from[j] of the batch.
struct split_t {
    std::vector<int64_t> from;
    int64_t first_rows = 0;
    std::vector<int32_t> first_qsl, rest_qsl;
    std::vector<uint8_t> rest_init;
};

split_t split_rows(const std::vector<int32_t> &lens,
        const std::vector<int32_t> &cuts, const std::vector<uint8_t> &init) {
    const std::vector<int32_t> qsl = offsets(lens);
    split_t sp;
    std::vector<int32_t> rest(lens.size());
    for (size_t i = 0; i < lens.size(); ++i) {
        for (int32_t t = 0; t < cuts[i]; ++t) {
            sp.from.push_back(qsl[i] + t);
        }
        rest[i] = lens[i] - cuts[i];
        sp.rest_init.push_back(cuts[i] > 0 ? uint8_t {1} : init[i]);
    }
    sp.first_rows = static_cast<int64_t>(sp.from.size());
    for (size_t i = 0; i < lens.size(); ++i) {
        for (int32_t t = cuts[i]; t < lens[i]; ++t) {
            sp.from.push_back(qsl[i] + t);
        }
    }
    sp.first_qsl = offsets(cuts);
    sp.rest_qsl = offsets(rest);
    return sp;
}

// x runs the batch as one call; y, holding x's rows in split order, as the
// two calls of sp. Each runs with its own params (algorithm, threads); call
// it in ASSERT_NO_FATAL_FAILURE.
void run_split(problem_t &x, problem_t &y, const rule_inputs_t *rx,
        const rule_inputs_t *ry, const std::vector<int32_t> &lens,
        const std::vector<uint8_t> &init, const split_t &sp,
        ng::gdn_kernel_t k) {
    ASSERT_EQ(call_rows(x, rx, 0, offsets(lens), init, k), status_t::success);
    ASSERT_EQ(call_rows(y, ry, 0, sp.first_qsl, init, k), status_t::success);
    ASSERT_EQ(call_rows(y, ry, sp.first_rows, sp.rest_qsl, sp.rest_init, k),
            status_t::success);
}

// y's output rows, row j holding row from[j] of x's: bit for bit when tol is
// 0, else within tol (absolute and relative). Only the HV x V values of a row
// count, and each must have been written (outputs start as NaN), except in
// x's rows listed in unwritten.
void expect_rows_match(const problem_t &x, const problem_t &y,
        const std::vector<int64_t> &from, float tol = 0.0f,
        const std::vector<int64_t> &unwritten = {}) {
    const size_t w = words(x.out);
    size_t bad = 0, nan = 0;
    int64_t first_bad = -1;
    float worst = 0.0f;
    for (size_t j = 0; j < from.size(); ++j) {
        if (std::find(unwritten.begin(), unwritten.end(), from[j])
                != unwritten.end()) {
            continue;
        }
        for (int64_t hv = 0; hv < x.s.HV; ++hv) {
            for (int64_t e = 0; e < x.s.V; ++e) {
                const size_t xi = static_cast<size_t>(
                        from[j] * x.p.out_stride_t + hv * x.p.out_stride_h + e);
                const size_t yi = static_cast<size_t>(
                        j * y.p.out_stride_t + hv * y.p.out_stride_h + e);
                const float a = x.out.get(xi), b = y.out.get(yi);
                const bool written = !std::isnan(a) && !std::isnan(b);
                const bool same = written
                        && (tol == 0.0f ? std::memcmp(x.out.raw.data() + xi * w,
                                                  y.out.raw.data() + yi * w,
                                                  w * sizeof(uint16_t))
                                                == 0
                                        : std::fabs(a - b)
                                                <= tol + tol * std::fabs(a));
                if (same) { continue; }
                ++bad;
                if (written) {
                    worst = std::max(worst, std::fabs(a - b));
                } else {
                    ++nan;
                }
                if (first_bad < 0) { first_bad = from[j]; }
            }
        }
    }
    EXPECT_EQ(bad, 0u) << "output: " << bad << " elements differ (" << nan
                       << " not written), the first in row " << first_bad
                       << ", worst abs diff " << worst;
}

// Raw 16-bit words in which two buffers differ.
size_t differing_words(const buf_t &x, const buf_t &y) {
    size_t n = 0;
    for (size_t i = 0; i < std::min(x.raw.size(), y.raw.size()); ++i) {
        n += x.raw[i] != y.raw[i];
    }
    return n + std::max(x.raw.size(), y.raw.size())
            - std::min(x.raw.size(), y.raw.size());
}

// Outputs (see expect_rows_match) and both pools: the SSM state bit for bit
// when its tolerances are 0, the conv state always.
void expect_split_matches(const problem_t &x, const problem_t &y,
        const std::vector<int64_t> &from, float otol = 0.0f,
        float stol_abs = 0.0f, float stol_rel = 0.0f,
        const std::vector<int64_t> &unwritten = {}) {
    expect_rows_match(x, y, from, otol, unwritten);
    if (stol_abs == 0.0f && stol_rel == 0.0f) {
        EXPECT_EQ(differing_words(x.ssm_state, y.ssm_state), 0u)
                << "ssm_state words differ";
    } else {
        expect_close(y.ssm_state, x.ssm_state, stol_abs, stol_rel, "ssm_state");
    }
    EXPECT_EQ(differing_words(x.conv_state, y.conv_state), 0u)
            << "conv_state words differ";
}

struct split_cfg_t {
    shape_t s;
    data_type_t src;
    ng::gdn_state_layout_t layout;
    bool norm, ds_conv;
};

testing::Message split_cfg_tag(const split_cfg_t &c) {
    return testing::Message()
            << c.s << " " << dt_tag(c.src) << (c.layout == kKv ? " kv" : " vk")
            << (c.norm ? " norm" : "") << (c.ds_conv ? " ds" : "");
}

// The batch of the split tests below: a long prompt, one starting from zero,
// three short ones and a pad slot.
const std::vector<int32_t> kSplitLens {300, 97, 64, 45, 130, 20};
const std::vector<uint8_t> kSplitInit {1, 0, 1, 1, 0, 1};

// x's rows of the sequences (lengths lens) on the pad slot, which no call
// writes.
std::vector<int64_t> pad_rows(
        const problem_t &x, const std::vector<int32_t> &lens) {
    const std::vector<int32_t> qsl = offsets(lens);
    std::vector<int64_t> rows;
    for (size_t i = 0; i < lens.size(); ++i) {
        if (x.idx[i] != x.p.pad_slot_id) { continue; }
        for (int32_t r = qsl[i]; r < qsl[i + 1]; ++r) {
            rows.push_back(r);
        }
    }
    return rows;
}

problem_t split_problem(const split_cfg_t &c, const split_t &sp,
        ng::gdn_prefill_algo_t algo, int32_t threads) {
    problem_t x(c.s, static_cast<int64_t>(sp.from.size()), 8, kF32, c.layout,
            c.norm, 73, c.src);
    if (c.ds_conv) { x.use_ds_conv_state(); }
    x.idx = distinct_slots(6, 8, 19);
    x.idx[5] = -1;
    x.p.prefill_algo = algo;
    x.p.num_threads = threads;
    return x;
}

// The chunked form continues a prefill with the bits of one call when the
// state is FP32 and the cut falls on one of the prompt's 32-token chunk
// boundaries: the chunks start at each call's first token. The two calls
// group the prompts differently and split heads into other value blocks,
// which changes no bit. Cuts after 3 chunks and after 1, at a prompt's end (no
// rest), at 0 (no first part), and before a 2-token rest, shorter than the
// conv history.
TEST(GdnSplitPass, ChunkedCutsOnChunkBoundariesMatchOneCall) {
    GDN_SKIP_UNLESS_NATIVE();
    const split_t sp
            = split_rows(kSplitLens, {96, 32, 64, 0, 128, 0}, kSplitInit);
    const split_cfg_t cfgs[] = {{kQwen36, kBf16, kKv, false, false},
            {kQwen36, kBf16, kVk, true, false},
            {kQwen38, kBf16, kKv, true, true},
            {kQwen35, kBf16, kKv, false, false},
            {kQwen36, kF16, kKv, true, false},
            {kQwen36, kF32, kVk, false, true},
            {kSmall, kBf16, kKv, true, true}};
    for (const split_cfg_t &c : cfgs) {
        for (int32_t threads : {1, 8, 64}) {
            SCOPED_TRACE(split_cfg_tag(c) << " T" << threads);
            problem_t x = split_problem(
                    c, sp, ng::gdn_prefill_algo_t::chunked, threads);
            // The split calls run at another thread count.
            problem_t y = x;
            y.p.num_threads = threads == 1 ? 64 : 1;
            gather_rows(x, y, sp.from);
            ASSERT_NO_FATAL_FAILURE(run_split(x, y, nullptr, nullptr,
                    kSplitLens, kSplitInit, sp, ng::gdn_kernel_t::native));
            expect_split_matches(
                    x, y, sp.from, 0.0f, 0.0f, 0.0f, pad_rows(x, kSplitLens));
        }
    }
}

// The token-recurrent form continues a prefill at any cut with the bits of
// one call, on the native and the reference kernel: cuts after 1 and 2 tokens
// (inside the conv history), on and off chunk boundaries, and at a prompt's
// end; the norm on the rounded core output and on the FP32 one.
TEST(GdnSplitPass, RecurrentCutsAnywhereMatchOneCall) {
    if (kernel_env_set() || prefill_env_set()) {
        GTEST_SKIP() << "ZENDNNL_GDN_KERNEL / ZENDNNL_GDN_PREFILL is set";
    }
    const std::vector<int32_t> lens {70, 33, 5, 9};
    const std::vector<uint8_t> init {1, 0, 1, 1};
    struct cfg_t {
        split_cfg_t c;
        bool round_core_out;
        ng::gdn_kernel_t k;
    };
    const auto native = ng::gdn_kernel_t::native;
    const auto ref = ng::gdn_kernel_t::reference;
    const cfg_t cfgs[] = {{{kQwen36, kBf16, kKv, true, false}, true, native},
            {{kQwen36, kBf16, kVk, true, false}, false, native},
            {{kQwen38, kF16, kKv, false, true}, true, native},
            {{kQwen36, kF32, kVk, false, false}, true, native},
            {{kSmall, kBf16, kKv, true, true}, true, ref},
            {{kSmall, kF32, kVk, false, false}, true, ref}};
    const std::vector<std::vector<int32_t>> cut_sets {
            {1, 1, 1, 1}, {17, 31, 2, 0}, {33, 32, 4, 9}, {69, 2, 3, 8}};
    for (const cfg_t &c : cfgs) {
        if (c.k == native && !has_native_isa()) { continue; }
        for (const std::vector<int32_t> &cuts : cut_sets) {
            for (int32_t threads : {1, 32}) {
                SCOPED_TRACE(split_cfg_tag(c.c)
                        << " " << ng::kernel_to_string(c.k)
                        << (c.round_core_out ? "" : " fp32 norm") << " cuts "
                        << cuts[0] << "," << cuts[1] << "," << cuts[2] << ","
                        << cuts[3] << " T" << threads);
                const split_t sp = split_rows(lens, cuts, init);
                problem_t x(c.c.s, static_cast<int64_t>(sp.from.size()), 6,
                        kF32, c.c.layout, c.c.norm, 79, c.c.src);
                if (c.c.ds_conv) { x.use_ds_conv_state(); }
                x.idx = distinct_slots(4, 6, 23);
                x.p.round_core_out = c.round_core_out;
                x.p.prefill_algo = ng::gdn_prefill_algo_t::recurrent;
                x.p.num_threads = threads;
                problem_t y = x;
                gather_rows(x, y, sp.from);
                ASSERT_NO_FATAL_FAILURE(
                        run_split(x, y, nullptr, nullptr, lens, init, sp, c.k));
                expect_split_matches(x, y, sp.from);
            }
        }
    }
}

// Off a chunk boundary the continuation's chunks start elsewhere, so the
// chunked form matches one call only to the precision of its operands (see
// chunked_tol).
TEST(GdnSplitPass, ChunkedCutsOffChunkBoundariesStayClose) {
    GDN_SKIP_UNLESS_NATIVE();
    const split_t sp
            = split_rows(kSplitLens, {40, 33, 50, 1, 100, 0}, kSplitInit);
    for (const split_cfg_t &c : {split_cfg_t {kQwen36, kBf16, kKv, true, false},
                 split_cfg_t {kQwen36, kF16, kVk, false, false},
                 split_cfg_t {kQwen36, kF32, kKv, true, true}}) {
        SCOPED_TRACE(split_cfg_tag(c));
        problem_t x = split_problem(c, sp, ng::gdn_prefill_algo_t::chunked, 0);
        problem_t y = x;
        gather_rows(x, y, sp.from);
        ASSERT_NO_FATAL_FAILURE(run_split(x, y, nullptr, nullptr, kSplitLens,
                kSplitInit, sp, ng::gdn_kernel_t::native));
        const tol_t t = chunked_tol(c.src, kF32, c.norm);
        expect_split_matches(x, y, sp.from, t.out, t.state_atol, t.state_rtol,
                pad_rows(x, kSplitLens));
    }
}

// Prompts continuing read-only from one slot (update_state = false) after a
// call prefilled their shared context into it give each prompt the bits of one
// call over the context and that prompt, on packed rows with the conv and on
// chunk_gated_delta_rule inputs. The context ends on a chunk boundary for the
// chunked form and off one for the recurrent form.
TEST(GdnSplitPass, PromptsFromOneSharedStateMatchOneCall) {
    if (kernel_env_set() || prefill_env_set()) {
        GTEST_SKIP() << "ZENDNNL_GDN_KERNEL / ZENDNNL_GDN_PREFILL is set";
    }
    const std::vector<int32_t> plens {1, 31, 50, 96, 181};
    const int64_t n = static_cast<int64_t>(plens.size());
    const std::vector<int32_t> pqsl = offsets(plens);
    for (ng::gdn_kernel_t k : kernels_to_check()) {
        const bool native = k == ng::gdn_kernel_t::native;
        for (bool rule : {false, true}) {
            for (ng::gdn_prefill_algo_t algo :
                    {ng::gdn_prefill_algo_t::recurrent,
                            ng::gdn_prefill_algo_t::chunked}) {
                if (!native && algo == ng::gdn_prefill_algo_t::chunked) {
                    continue;
                }
                for (data_type_t src : {kBf16, kF32}) {
                    const shape_t s = native
                            ? (rule ? shape_t {16, 32, 128, 128, 0} : kQwen36)
                            : (rule ? shape_t {2, 4, 64, 32, 0} : kSmall);
                    const int32_t ctx
                            = algo == ng::gdn_prefill_algo_t::chunked ? 96 : 70;
                    SCOPED_TRACE(testing::Message()
                            << ng::kernel_to_string(k) << " "
                            << ng::prefill_algo_to_string(algo) << " " << s
                            << " " << dt_tag(src)
                            << (rule ? " rule inputs" : " packed"));
                    problem_t y(
                            s, ctx + pqsl.back(), 2, kF32, kKv, true, 83, src);
                    y.p.prefill_algo = algo;
                    // Prompt j of x: the context rows of y, then its own.
                    std::vector<int64_t> from;
                    std::vector<int32_t> xlens;
                    for (int64_t j = 0; j < n; ++j) {
                        for (int32_t t = 0; t < ctx; ++t) {
                            from.push_back(t);
                        }
                        for (int32_t t = 0; t < plens[j]; ++t) {
                            from.push_back(ctx + pqsl[j] + t);
                        }
                        xlens.push_back(ctx + plens[j]);
                    }
                    problem_t x(s, static_cast<int64_t>(from.size()), n, kF32,
                            kKv, true, 83, src);
                    x.a_log = y.a_log;
                    x.dt_bias = y.dt_bias;
                    x.conv_w = y.conv_w;
                    x.conv_b = y.conv_b;
                    x.norm_w = y.norm_w;
                    x.p.prefill_algo = algo;
                    x.idx = distinct_slots(n, n, 29);
                    gather_rows(y, x, from);
                    rule_inputs_t ry, rx;
                    if (rule) {
                        ry = rule_inputs(y, kF32, kF32);
                        rx = rule_inputs(x, kF32, kF32);
                    }
                    y.idx = {1};
                    ASSERT_EQ(call_rows(y, rule ? &ry : nullptr, 0, {0, ctx},
                                      {0}, k),
                            status_t::success);
                    y.idx.assign(static_cast<size_t>(n), 1);
                    y.p.update_state = false;
                    ASSERT_EQ(call_rows(y, rule ? &ry : nullptr, ctx, pqsl,
                                      std::vector<uint8_t>(n, 1), k),
                            status_t::success);
                    ASSERT_EQ(call_rows(x, rule ? &rx : nullptr, 0,
                                      offsets(xlens),
                                      std::vector<uint8_t>(n, 0), k),
                            status_t::success);
                    expect_rows_match(y, x, from);
                }
            }
        }
    }
}

// Speculative verify: draft tokens checked from the state the previous step
// accepted, with a state stored after every token.

// n sequences of draft tokens in x's rows; sequence i owns SSM slots
// spec[i * stride .. (i + 1) * stride) and conv slot spec[i * stride], whose
// conv state has conv_taps() taps (DS or SD).
struct verify_t {
    problem_t x;
    int64_t stride;
    std::vector<int32_t> spec, accepted;
};

verify_t make_verify(const split_cfg_t &c, data_type_t state_dt, int64_t n,
        int64_t tokens, int64_t stride, uint32_t seed) {
    const int64_t slots = n * stride + 1;
    verify_t v {problem_t(c.s, tokens, slots, state_dt, c.layout, c.norm, seed,
                        c.src),
            stride, distinct_slots(n * stride, slots, seed + 1), {}};
    problem_t &x = v.x;
    const int64_t D = 2 * c.s.H * c.s.K + c.s.HV * c.s.V;
    const int64_t L = c.s.W > 1 ? c.s.W - 1 + stride - 1 : 0;
    x.conv_state = buf_t(c.src, static_cast<size_t>(slots * L * D + 1));
    std::mt19937 rng(seed + 2);
    x.conv_state.fill_uniform(rng, -1.0f, 1.0f);
    x.p.conv_state_slot_stride = L * D;
    x.p.conv_state_tap_stride = c.ds_conv ? 1 : D;
    x.p.conv_state_dim_stride = c.ds_conv ? L : 1;
    x.p.spec_state_stride = stride;
    x.idx.assign(static_cast<size_t>(n), 0);
    return v;
}

// A verify call on rows [r0, r0 + qsl.back()) of v.
status_t verify_rows(verify_t &v, int64_t r0, const std::vector<int32_t> &qsl,
        ng::gdn_kernel_t k) {
    v.x.p.spec_state_indices = v.spec.data();
    v.x.p.num_accepted_tokens = v.accepted.data();
    return call_rows(
            v.x, nullptr, r0, qsl, std::vector<uint8_t>(qsl.size() - 1, 1), k);
}

// The decode twin of v, before its first verify call: slot i holds sequence
// i's state from slot spec[i * stride + accepted[i] - 1], conv slot i the
// conv_width - 1 taps from tap accepted[i] - 1, and row s * n + i the token
// that sequence i decodes in step s (rows[s * n + i] of v, or -1: a pad slot
// in that step). With verify_layout its conv slots have v's strides.
problem_t decode_twin(const verify_t &v, const split_cfg_t &c,
        const std::vector<int64_t> &rows, int64_t n,
        bool verify_layout = false) {
    const problem_t &x = v.x;
    problem_t tw(c.s, static_cast<int64_t>(rows.size()), n, x.p.state_dt,
            c.layout, c.norm, 103, c.src);
    if (verify_layout) {
        tw.p.conv_state_slot_stride = x.p.conv_state_slot_stride;
        tw.p.conv_state_tap_stride = x.p.conv_state_tap_stride;
        tw.p.conv_state_dim_stride = x.p.conv_state_dim_stride;
        tw.conv_state = buf_t(
                c.src, static_cast<size_t>(n * x.p.conv_state_slot_stride + 1));
    } else if (c.ds_conv) {
        tw.use_ds_conv_state();
    }
    tw.a_log = x.a_log;
    tw.dt_bias = x.dt_bias;
    tw.conv_w = x.conv_w;
    tw.conv_b = x.conv_b;
    tw.norm_w = x.norm_w;
    tw.p.num_threads = x.p.num_threads;
    tw.p.allow_neg_eigval = x.p.allow_neg_eigval;
    std::vector<int64_t> from(rows.size());
    for (size_t j = 0; j < rows.size(); ++j) {
        from[j] = std::max<int64_t>(rows[j], 0);
    }
    gather_rows(x, tw, from);
    const size_t sw = static_cast<size_t>(x.p.ssm_state_slot_stride)
            * words(x.ssm_state);
    const int64_t D = 2 * c.s.H * c.s.K + c.s.HV * c.s.V;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t acc = v.accepted[i];
        std::memcpy(tw.ssm_state.raw.data() + i * sw,
                x.ssm_state.raw.data() + v.spec[i * v.stride + acc - 1] * sw,
                sw * sizeof(uint16_t));
        for (int64_t j = 0; j + 1 < c.s.W; ++j) {
            for (int64_t ch = 0; ch < D; ++ch) {
                tw.conv_state.set(i * tw.p.conv_state_slot_stride
                                + j * tw.p.conv_state_tap_stride
                                + ch * tw.p.conv_state_dim_stride,
                        x.conv_state.get(v.spec[i * v.stride]
                                        * x.p.conv_state_slot_stride
                                + (acc - 1 + j) * x.p.conv_state_tap_stride
                                + ch * x.p.conv_state_dim_stride));
            }
        }
    }
    return tw;
}

// Decode step s of the twin: sequence i decodes row s * n + i on slot i, or
// sits out as a pad slot.
status_t twin_step(problem_t &tw, const std::vector<int64_t> &rows, int64_t s,
        int64_t n, ng::gdn_kernel_t k) {
    tw.idx.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        tw.idx[i] = rows[s * n + i] < 0 ? -1 : static_cast<int32_t>(i);
    }
    return call_rows(tw, nullptr, s * n,
            offsets(std::vector<int32_t>(static_cast<size_t>(n), 1)),
            std::vector<uint8_t>(static_cast<size_t>(n), 1), k, n);
}

bool same_slot(const problem_t &x, int64_t xs, const problem_t &y, int64_t ys) {
    const size_t n = static_cast<size_t>(x.p.ssm_state_slot_stride)
            * words(x.ssm_state);
    return std::memcmp(x.ssm_state.raw.data() + xs * n,
                   y.ssm_state.raw.data() + ys * n, n * sizeof(uint16_t))
            == 0;
}

// Each verified sequence's conv state after the call: the conv_width - 2
// taps from tap accepted[i] on (of `before`), its q tokens, then the taps of
// `before` that follow.
void expect_conv_rolled(const verify_t &v, const buf_t &before,
        const std::vector<int32_t> &qsl, int64_t r0) {
    const problem_t &x = v.x;
    const int64_t D = 2 * x.s.H * x.s.K + x.s.HV * x.s.V;
    const int64_t L = x.s.W > 1 ? x.s.W - 1 + v.stride - 1 : 0;
    const int64_t W2 = x.s.W - 2;
    const auto at = [&x](int64_t slot, int64_t tap, int64_t ch) {
        return static_cast<size_t>(slot * x.p.conv_state_slot_stride
                + tap * x.p.conv_state_tap_stride
                + ch * x.p.conv_state_dim_stride);
    };
    size_t bad = 0;
    for (size_t i = 0; i + 1 < qsl.size(); ++i) {
        const int64_t q = qsl[i + 1] - qsl[i];
        const int64_t slot = v.spec[i * v.stride];
        if (q == 0 || slot < 0) { continue; }
        for (int64_t ch = 0; ch < D; ++ch) {
            for (int64_t j = 0; j < L; ++j) {
                const float want = j < W2
                        ? before.get(at(slot, v.accepted[i] + j, ch))
                        : j < W2 + q
                        ? x.qkv.get(
                                  (r0 + qsl[i] + j - W2) * x.p.qkv_stride + ch)
                        : before.get(at(slot, j, ch));
                bad += x.conv_state.get(at(slot, j, ch)) != want;
            }
        }
    }
    EXPECT_EQ(bad, 0u) << "conv state taps out of place";
}

struct verify_cfg_t {
    split_cfg_t c;
    data_type_t state_dt;
    bool neg_eigval = false;
};

const verify_cfg_t kVerifyCfgs[] = {{{kQwen36, kBf16, kKv, false, false}, kF32},
        {{kQwen36, kBf16, kVk, true, true}, kF32},
        {{kQwen38, kF16, kKv, true, false}, kBf16},
        {{kQwen36, kF32, kKv, false, true}, kF16},
        {{kSmall, kBf16, kKv, true, false}, kF32},
        {{kQwen36, kBf16, kKv, true, false}, kF32, true}};

std::vector<int32_t> verify_threads(ng::gdn_kernel_t k) {
    return k == ng::gdn_kernel_t::native ? std::vector<int32_t> {1, 64}
                                         : std::vector<int32_t> {0};
}

// One verify call over v's sequences (lengths lens) against decode steps from
// the accepted states on its twin: every output, the state stored after every
// token, and the conv state afterwards. With verify_layout the twin's conv
// state has v's strides.
void expect_verify_matches_decode(verify_t &v, const split_cfg_t &c,
        const std::vector<int32_t> &lens, ng::gdn_kernel_t k,
        bool verify_layout = false) {
    const std::vector<int32_t> qsl = offsets(lens);
    const int64_t n = static_cast<int64_t>(lens.size()), stride = v.stride;
    std::vector<int64_t> rows(static_cast<size_t>(stride * n));
    for (int64_t s = 0; s < stride; ++s) {
        for (int64_t i = 0; i < n; ++i) {
            rows[s * n + i] = s < lens[i] ? qsl[i] + s : -1;
        }
    }
    problem_t tw = decode_twin(v, c, rows, n, verify_layout);
    const buf_t before = v.x.conv_state;
    ASSERT_EQ(verify_rows(v, 0, qsl, k), status_t::success);
    for (int64_t s = 0; s < stride; ++s) {
        ASSERT_EQ(twin_step(tw, rows, s, n, k), status_t::success);
        for (int64_t i = 0; i < n; ++i) {
            if (s >= lens[i]) { continue; }
            EXPECT_TRUE(same_slot(v.x, v.spec[i * stride + s], tw, i))
                    << "sequence " << i << ", token " << s;
        }
    }
    std::vector<int64_t> to_twin(static_cast<size_t>(qsl.back()));
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t t = 0; t < lens[i]; ++t) {
            to_twin[qsl[i] + t] = t * n + i;
        }
    }
    expect_rows_match(tw, v.x, to_twin);
    expect_conv_rolled(v, before, qsl, 0);
}

// One verify call gives, bit for bit, what that many decode steps from the
// accepted state give. Sequences of 16, 9 and 2 draft tokens (the shorter
// ones sit out the later decode steps as pad slots), accepted counts of 1 and
// of every token, every source and state type, kv and vk, the fused norm, and
// DS and SD conv states.
TEST(GdnSpecVerify, MatchesDecodeSteps) {
    if (kernel_env_set()) { GTEST_SKIP() << "ZENDNNL_GDN_KERNEL is set"; }
    const std::vector<int32_t> lens {16, 9, 2};
    for (ng::gdn_kernel_t k : kernels_to_check()) {
        for (const verify_cfg_t &c : kVerifyCfgs) {
            for (int32_t threads : verify_threads(k)) {
                SCOPED_TRACE(split_cfg_tag(c.c)
                        << " " << dt_tag(c.state_dt) << " state "
                        << (c.neg_eigval ? "neg_eigval " : "")
                        << ng::kernel_to_string(k) << " T" << threads);
                verify_t v = make_verify(c.c, c.state_dt, 3, 27, 16, 107);
                v.accepted = {1, 9, 2};
                v.x.p.allow_neg_eigval = c.neg_eigval;
                v.x.p.num_threads = threads;
                ASSERT_NO_FATAL_FAILURE(
                        expect_verify_matches_decode(v, c.c, lens, k));
            }
        }
    }
}

// Verify gives the bits of decode steps on a conv state of the same strides
// whatever its layout: vLLM's DS buffer with draft taps decoded in place, a
// padded DS layout of one tap (both read as DS), and channels at a stride of 2
// in an SD-like layout (read in scalar arithmetic). f32 and bf16 sources, with
// SiLU.
TEST(GdnSpecVerify, MatchesDecodeOnAnyConvLayout) {
    if (kernel_env_set()) { GTEST_SKIP() << "ZENDNNL_GDN_KERNEL is set"; }
    const int64_t D = 2 * kSmall.H * kSmall.K + kSmall.HV * kSmall.V;
    for (ng::gdn_kernel_t k : kernels_to_check()) {
        for (data_type_t src : {kF32, kBf16}) {
            SCOPED_TRACE(testing::Message()
                    << ng::kernel_to_string(k) << " " << dt_tag(src));
            const split_cfg_t ds {kSmall, src, kKv, false, true};
            const split_cfg_t sd {kSmall, src, kKv, false, false};
            const split_cfg_t one_tap {
                    shape_t {2, 4, 64, 32, 2}, src, kKv, false, true};
            {
                SCOPED_TRACE("DS with draft taps");
                verify_t v = make_verify(ds, kF32, 3, 14, 8, 137);
                v.accepted = {1, 8, 3};
                ASSERT_NO_FATAL_FAILURE(expect_verify_matches_decode(
                        v, ds, {8, 5, 1}, k, true));
            }
            {
                SCOPED_TRACE("padded DS, one tap");
                verify_t v = make_verify(one_tap, kF32, 2, 2, 1, 139);
                set_conv_strides(v.x, 1, 1, 2, 141);
                v.accepted = {1, 1};
                ASSERT_NO_FATAL_FAILURE(expect_verify_matches_decode(
                        v, one_tap, {1, 1}, k, true));
            }
            {
                SCOPED_TRACE("SD, channel stride 2");
                verify_t v = make_verify(sd, kF32, 3, 14, 8, 143);
                set_conv_strides(v.x, kSmall.W - 1 + 8 - 1, 2 * D, 2, 145);
                v.accepted = {1, 8, 3};
                ASSERT_NO_FATAL_FAILURE(expect_verify_matches_decode(
                        v, sd, {8, 5, 1}, k, true));
            }
        }
    }
}

// Two verify steps of n sequences at stride 8: lens1 tokens with acc1
// accepted, then lens2 tokens with acc2 accepted (acc2[i] <= lens1[i]).
struct next_step_case_t {
    std::vector<int32_t> lens1, acc1, lens2, acc2;
};

// Full first steps; a short first step; then a row that drafted nothing and
// rows shorter than the count the step before them accepted.
const next_step_case_t kNextStepCases[]
        = {{{8, 8, 8}, {1, 4, 8}, {8, 8, 8}, {3, 8, 1}},
                {{5, 2, 8}, {1, 2, 8}, {8, 8, 8}, {3, 2, 5}},
                {{8, 8, 8}, {1, 1, 1}, {1, 3, 8}, {8, 6, 2}}};

// The second step gives what decode steps over the first step's tokens that
// it accepts and then over its own tokens give: every output, the state
// stored after every token and the conv state afterwards.
void expect_next_step(const verify_cfg_t &c, const shape_t &s,
        ng::gdn_kernel_t k, const next_step_case_t &nc) {
    const int64_t n = static_cast<int64_t>(nc.lens1.size()), stride = 8;
    const std::vector<int32_t> qsl1 = offsets(nc.lens1);
    const std::vector<int32_t> qsl2 = offsets(nc.lens2);
    const int64_t T1 = qsl1.back(), T2 = qsl2.back();
    split_cfg_t sc = c.c;
    sc.s = s;
    verify_t v = make_verify(sc, c.state_dt, n, T1 + T2, stride, 109);
    v.accepted = nc.acc1;
    v.x.p.allow_neg_eigval = c.neg_eigval;
    // Steps 0 .. stride - 1 decode the first step's tokens that the second
    // accepts, steps stride .. 2 * stride - 1 the second step's tokens.
    std::vector<int64_t> rows(static_cast<size_t>(2 * stride * n), -1);
    for (int64_t t = 0; t < stride; ++t) {
        for (int64_t i = 0; i < n; ++i) {
            if (t < nc.acc2[i]) { rows[t * n + i] = qsl1[i] + t; }
            if (t < nc.lens2[i]) {
                rows[(stride + t) * n + i] = T1 + qsl2[i] + t;
            }
        }
    }
    problem_t tw = decode_twin(v, sc, rows, n);
    ASSERT_EQ(verify_rows(v, 0, qsl1, k), status_t::success);
    const buf_t after_first = v.x.conv_state;
    v.accepted = nc.acc2;
    ASSERT_EQ(verify_rows(v, T1, qsl2, k), status_t::success);
    for (int64_t t = 0; t < 2 * stride; ++t) {
        ASSERT_EQ(twin_step(tw, rows, t, n, k), status_t::success);
        for (int64_t i = 0; t >= stride && i < n; ++i) {
            if (t - stride >= nc.lens2[i]) { continue; }
            EXPECT_TRUE(same_slot(v.x, v.spec[i * stride + t - stride], tw, i))
                    << "sequence " << i << ", token " << t - stride;
        }
    }
    std::vector<int64_t> second_rows, to_twin;
    for (int64_t r = 0; r < T2; ++r) {
        second_rows.push_back(T1 + r);
    }
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t t = 0; t < nc.lens2[i]; ++t) {
            to_twin.push_back((stride + t) * n + i);
        }
    }
    problem_t second = v.x;
    gather_rows(v.x, second, second_rows);
    expect_rows_match(tw, second, to_twin);
    expect_conv_rolled(v, after_first, qsl2, T1);
}

// A verify step continues from the state after the token the step before it
// accepted last, whatever either step's length: with full and short first
// steps, a row that drafted nothing after a step that accepted all its tokens,
// and conv widths 4, 2 (no history taps kept) and 0.
TEST(GdnSpecVerify, NextStepContinuesFromTheAcceptedToken) {
    if (kernel_env_set()) { GTEST_SKIP() << "ZENDNNL_GDN_KERNEL is set"; }
    for (ng::gdn_kernel_t k : kernels_to_check()) {
        for (const verify_cfg_t &c : kVerifyCfgs) {
            for (int64_t w : {c.c.s.W, int64_t {2}, int64_t {0}}) {
                shape_t s = c.c.s;
                s.W = w;
                for (size_t ci = 0;
                        ci < sizeof(kNextStepCases) / sizeof(kNextStepCases[0]);
                        ++ci) {
                    SCOPED_TRACE(split_cfg_tag(c.c)
                            << " " << dt_tag(c.state_dt) << " state "
                            << (c.neg_eigval ? "neg_eigval " : "")
                            << ng::kernel_to_string(k) << " W" << w << " case "
                            << ci);
                    expect_next_step(c, s, k, kNextStepCases[ci]);
                }
            }
        }
    }
}

// A sequence whose first slot is the pad slot writes zero output rows and
// touches neither pool, and an empty sequence touches nothing.
TEST(GdnSpecVerify, PadAndEmptySequencesAreSkipped) {
    if (kernel_env_set()) { GTEST_SKIP() << "ZENDNNL_GDN_KERNEL is set"; }
    const std::vector<int32_t> qsl = offsets({4, 0, 3});
    const int64_t stride = 4;
    for (ng::gdn_kernel_t k : kernels_to_check()) {
        for (bool norm : {false, true}) {
            SCOPED_TRACE(std::string(ng::kernel_to_string(k))
                    + (norm ? " norm" : ""));
            verify_t v = make_verify({kSmall, kBf16, kKv, norm, false}, kF32, 3,
                    qsl.back(), stride, 113);
            const std::vector<int32_t> slots = v.spec;
            v.spec[0] = -1;
            v.accepted = {1, 1, 2};
            const problem_t before = v.x;
            ASSERT_EQ(verify_rows(v, 0, qsl, k), status_t::success);
            const int64_t st = v.x.p.out_stride_t;
            for (int64_t e = qsl[0] * st; e < qsl[1] * st; ++e) {
                ASSERT_EQ(v.x.out.get(e), 0.0f) << "pad row element " << e;
            }
            for (int64_t e = qsl[2] * st; e < qsl[3] * st; ++e) {
                ASSERT_FALSE(std::isnan(v.x.out.get(e)))
                        << "third sequence, element " << e;
            }
            for (int64_t t = 0; t < 2 * stride; ++t) {
                EXPECT_TRUE(same_slot(v.x, slots[t], before, slots[t]))
                        << "SSM slot " << slots[t];
            }
            const size_t cw = static_cast<size_t>(v.x.p.conv_state_slot_stride)
                    * words(v.x.conv_state);
            for (const int32_t s : {slots[0], slots[stride]}) {
                EXPECT_EQ(std::memcmp(v.x.conv_state.raw.data() + s * cw,
                                  before.conv_state.raw.data() + s * cw,
                                  cw * sizeof(uint16_t)),
                        0)
                        << "conv slot " << s;
            }
        }
    }
}

// Speculative verify takes no decodes, update_state = true, packed rows and
// raw gates, conv_width 0 or at least 2, cu_seqlens, at most
// spec_state_stride tokens per sequence, 1 <= num_accepted_tokens <=
// spec_state_stride and a slot in every column it reads or writes; a refused
// call touches no buffer. Each tweak passes every other rule, so it fails on
// its own.
TEST(GdnApi, RejectsInvalidSpecVerify) {
    GDN_SKIP_UNLESS_VALIDATING();
    const std::vector<int32_t> qsl = offsets({3, 2});
    using tweak_t = void (*)(verify_t &, ng::gdn_params &);
    const tweak_t tweaks[] = {
            [](verify_t &v, ng::gdn_params &) { v.accepted[0] = 0; },
            [](verify_t &v, ng::gdn_params &) { v.accepted[1] = 5; },
            [](verify_t &v, ng::gdn_params &) {
        // Sequence 1 continues from the slot in column 3, past its tokens.
        v.accepted[1] = 4;
        v.spec[7] = -1;
    },
            [](verify_t &, ng::gdn_params &q) { q.spec_state_stride = 2; },
            [](verify_t &, ng::gdn_params &q) { q.spec_state_stride = 0; },
            [](verify_t &, ng::gdn_params &q) {
        q.num_accepted_tokens = nullptr;
    },
            [](verify_t &, ng::gdn_params &q) {
        static const std::vector<int32_t> one_decode {0, 1, 3};
        q.cu_seqlens = one_decode.data();
        q.num_tokens = 3;
        q.num_decodes = 1;
    },
            [](verify_t &, ng::gdn_params &q) { q.update_state = false; },
            [](verify_t &, ng::gdn_params &q) {
        q.gate_input = ng::gdn_gate_input_t::precomputed;
    },
            [](verify_t &v, ng::gdn_params &q) {
        q.k = v.x.qkv.data();
        q.v = v.x.qkv.data();
        q.conv_width = 0;
        q.q_stride_h = q.k_stride_h = q.head_k_dim;
        q.k_stride_t = q.num_k_heads * q.head_k_dim;
        q.v_stride_h = q.head_v_dim;
        q.v_stride_t = q.num_v_heads * q.head_v_dim;
    },
            [](verify_t &, ng::gdn_params &q) { q.conv_width = 1; },
            [](verify_t &, ng::gdn_params &q) { q.num_state_slots = 2; },
            [](verify_t &v, ng::gdn_params &) { v.spec[1] = -1; },
            [](verify_t &, ng::gdn_params &q) {
        q.cu_seqlens = nullptr;
        q.num_tokens = q.batch;
    },
            [](verify_t &, ng::gdn_params &q) {
        q.spec_state_indices = nullptr;
    },
    };
    // Run 0 leaves the call valid; run i applies tweak i - 1.
    for (size_t i = 0; i <= sizeof(tweaks) / sizeof(tweaks[0]); ++i) {
        SCOPED_TRACE(testing::Message() << "run " << i);
        verify_t v = make_verify({kSmall, kBf16, kKv, false, false}, kF32, 2,
                qsl.back(), 4, 127);
        v.accepted = {1, 2};
        const problem_t before = v.x;
        ng::gdn_params q = v.x.p;
        q.batch = 2;
        q.num_tokens = qsl.back();
        q.cu_seqlens = qsl.data();
        q.spec_state_indices = v.spec.data();
        q.num_accepted_tokens = v.accepted.data();
        q.kernel = ng::gdn_kernel_t::reference;
        if (i > 0) { tweaks[i - 1](v, q); }
        const status_t st = ng::gdn_direct(v.x.qkv.data(), v.x.a.data(),
                v.x.b_ptr(), v.x.a_log.data(), v.x.dt_bias.data(),
                v.x.conv_w.data(), v.x.conv_b.data(), v.x.conv_state.data(),
                v.x.ssm_state.data(), nullptr, v.x.out.data(), q);
        if (i == 0) {
            EXPECT_EQ(st, status_t::success);
            continue;
        }
        EXPECT_EQ(st, status_t::failure);
        EXPECT_EQ(differing_words(v.x.out, before.out), 0u) << "output";
        EXPECT_EQ(differing_words(v.x.ssm_state, before.ssm_state), 0u)
                << "ssm_state";
        EXPECT_EQ(differing_words(v.x.conv_state, before.conv_state), 0u)
                << "conv_state";
    }
}

// The rule inputs take prefill-only calls without conv, consistent strides,
// and A_log / dt_bias only with raw gates.
TEST(GdnApi, RejectsInvalidRuleInputs) {
    GDN_SKIP_UNLESS_VALIDATING();
    const shape_t s {2, 4, 64, 32, 0};
    const std::vector<int32_t> qsl = offsets({5, 3});
    const std::vector<uint8_t> init {1, 0};
    using tweak_t = void (*)(ng::gdn_params &);
    const tweak_t tweaks[] = {
            [](ng::gdn_params &q) { q.conv_width = 4; },
            [](ng::gdn_params &q) { q.v = nullptr; },
            [](ng::gdn_params &q) { q.k_stride_h = q.head_k_dim - 16; },
            [](ng::gdn_params &q) { q.v_stride_t = q.v_stride_h; },
            [](ng::gdn_params &q) { q.a_dt = data_type_t::s8; },
            [](ng::gdn_params &q) {
        q.gate_input = static_cast<ng::gdn_gate_input_t>(3);
    },
            [](ng::gdn_params &q) { q.allow_neg_eigval = true; },
    };
    for (size_t i = 0; i < sizeof(tweaks) / sizeof(tweaks[0]); ++i) {
        SCOPED_TRACE(testing::Message() << "tweak " << i);
        problem_t y(s, qsl.back(), 3, kF32, kKv, false, 1);
        y.idx = {2, 0};
        rule_inputs_t r = rule_inputs(y, kF32, kF32);
        ng::gdn_params base = y.p;
        // Apply the tweak to a copy of what rule_call builds.
        ng::gdn_params q = base;
        q.batch = 2;
        q.num_tokens = qsl.back();
        q.cu_seqlens = qsl.data();
        q.has_initial_state = reinterpret_cast<const bool *>(init.data());
        q.qkv_stride = r.qs_t;
        q.q_stride_h = r.qs_h;
        q.k = r.k.data();
        q.k_stride_t = r.ks_t;
        q.k_stride_h = r.ks_h;
        q.v = r.v.data();
        q.v_stride_t = r.vs_t;
        q.v_stride_h = r.vs_h;
        q.gate_input = ng::gdn_gate_input_t::precomputed;
        q.a_dt = kF32;
        q.b_dt = kF32;
        q.a_stride = s.HV;
        q.b_stride = s.HV;
        tweaks[i](q);
        EXPECT_EQ(ng::gdn_direct(r.q.data(), r.g.data(), r.beta.data(), nullptr,
                          nullptr, y.conv_w.data(), y.conv_b.data(),
                          y.conv_state.data(), y.ssm_state.data(), y.idx.data(),
                          y.out.data(), q),
                status_t::failure);
    }
    // Decodes take neither separate q / k / v nor precomputed gates, and raw
    // gates need A_log and dt_bias.
    problem_t y(s, 2, 3, kF32, kKv, false, 1);
    y.idx = {2, 0};
    rule_inputs_t r = rule_inputs(y, kF32, kF32);
    ng::gdn_params q = y.p;
    q.batch = 2;
    q.num_decodes = 2;
    q.num_tokens = 2;
    q.gate_input = ng::gdn_gate_input_t::precomputed;
    q.a_stride = s.HV;
    q.b_stride = s.HV;
    EXPECT_EQ(ng::gdn_direct(y.qkv.data(), r.g.data(), r.beta.data(), nullptr,
                      nullptr, y.conv_w.data(), y.conv_b.data(),
                      y.conv_state.data(), y.ssm_state.data(), y.idx.data(),
                      y.out.data(), q),
            status_t::failure);
    q.num_decodes = 0;
    q.gate_input = ng::gdn_gate_input_t::raw;
    EXPECT_EQ(ng::gdn_direct(y.qkv.data(), y.a.data(), y.b_ptr(), nullptr,
                      nullptr, y.conv_w.data(), y.conv_b.data(),
                      y.conv_state.data(), y.ssm_state.data(), y.idx.data(),
                      y.out.data(), q),
            status_t::failure);
}

TEST(GdnApi, BatchZeroIsNoop) {
    problem_t x(kSmall, 1, 1, data_type_t::f32, ng::gdn_state_layout_t::kv,
            false, 1);
    x.idx = {0};
    x.p.batch = 0;
    const problem_t before = x;
    ASSERT_EQ(x.decode(ng::gdn_kernel_t::none), status_t::success);
    expect_same(x, const_cast<problem_t &>(before), 0.0f, "decode");
}

TEST(GdnApi, RejectsInvalidParameters) {
    GDN_SKIP_UNLESS_VALIDATING();
    using tweak_t = void (*)(problem_t &);
    const tweak_t tweaks[] = {
            [](problem_t &x) { x.p.num_threads = -1; },
            [](problem_t &x) { x.p.num_k_heads = 0; },
            [](problem_t &x) { x.p.num_v_heads = 3; },
            [](problem_t &x) { x.p.conv_width = -1; },
            [](problem_t &x) { x.p.src_dt = data_type_t::s8; },
            [](problem_t &x) {
        x.p.state_layout = static_cast<ng::gdn_state_layout_t>(7);
    },
            [](problem_t &x) { x.p.qkv_stride -= 17; },
            [](problem_t &x) { x.p.out_stride_t -= 1; },
            [](problem_t &x) { x.p.ssm_state_head_stride -= 1; },
            [](problem_t &x) { x.p.ssm_state_slot_stride = 0; },
            [](problem_t &x) { x.p.conv_state_tap_stride = 0; },
            [](problem_t &x) { x.p.conv_state_slot_stride -= 1; },
            [](problem_t &x) { x.p.num_state_slots = 1; },
            [](problem_t &x) {
        x.p.norm_weight = x.norm_w.data();
        x.p.norm_weight_dt = data_type_t::f32;
        x.p.z = x.z.data();
        x.p.z_stride_t = x.s.HV * x.s.V;
        x.p.z_stride_h = x.s.V;
        x.p.norm_gate = static_cast<ng::gdn_norm_gate_t>(5);
    },
            [](problem_t &x) {
        x.p.norm_weight = x.norm_w.data();
        x.p.norm_weight_dt = data_type_t::f32;
    },
    };
    for (size_t i = 0; i < sizeof(tweaks) / sizeof(tweaks[0]); ++i) {
        SCOPED_TRACE(testing::Message() << "tweak " << i);
        problem_t x(kSmall, 2, 3, data_type_t::f32, ng::gdn_state_layout_t::kv,
                false, 1);
        x.idx = {2, 0};
        tweaks[i](x);
        EXPECT_EQ(x.decode(ng::gdn_kernel_t::none), status_t::failure);
        EXPECT_EQ(x.prefill(ng::gdn_kernel_t::none, {0, 1, 2}, {1, 1}),
                status_t::failure);
    }

    problem_t x(kSmall, 4, 3, data_type_t::f32, ng::gdn_state_layout_t::kv,
            false, 1);
    x.idx = {2, 0};
    EXPECT_EQ(x.prefill(ng::gdn_kernel_t::none, {0, 3, 1}, {1, 1}),
            status_t::failure);
    EXPECT_EQ(x.prefill(ng::gdn_kernel_t::none, {0, 1, 5}, {1, 1}),
            status_t::failure);
    EXPECT_EQ(x.prefill(ng::gdn_kernel_t::none, {-1, 1, 2}, {1, 1}),
            status_t::failure);
    ng::gdn_params q = x.p;
    q.batch = 2;
    q.num_decodes = 2;
    q.num_tokens = 2;
    EXPECT_EQ(ng::gdn_direct(nullptr, x.a.data(), x.b_ptr(), x.a_log.data(),
                      x.dt_bias.data(), x.conv_w.data(), x.conv_b.data(),
                      x.conv_state.data(), x.ssm_state.data(), x.idx.data(),
                      x.out.data(), q),
            status_t::failure);
    EXPECT_EQ(ng::gdn_direct(x.qkv.data(), x.a.data(), x.b_ptr(),
                      x.a_log.data(), x.dt_bias.data(), x.conv_w.data(),
                      x.conv_b.data(), x.conv_state.data(), x.ssm_state.data(),
                      nullptr, x.out.data(), q),
            status_t::failure);
}

// update_state = false takes prefill-only calls, and a prefill that reads its
// slot still needs the pools.
TEST(GdnApi, RejectsUpdateStateFalseMisuse) {
    GDN_SKIP_UNLESS_VALIDATING();
    problem_t x(kSmall, 3, 3, kF32, kKv, false, 1);
    x.idx = {2, 0, 1};
    x.p.update_state = false;
    const problem_t before = x;
    EXPECT_EQ(x.decode(ng::gdn_kernel_t::none), status_t::failure);
    EXPECT_EQ(x.mixed(ng::gdn_kernel_t::none, 1, {0, 1, 2, 3}, {1, 1, 1}),
            status_t::failure);
    const std::vector<uint8_t> init {0, 1, 0};
    ng::gdn_params q = x.p;
    q.batch = 3;
    q.num_tokens = 3;
    q.has_initial_state = reinterpret_cast<const bool *>(init.data());
    EXPECT_EQ(
            ng::gdn_direct(x.qkv.data(), x.a.data(), x.b_ptr(), x.a_log.data(),
                    x.dt_bias.data(), x.conv_w.data(), x.conv_b.data(), nullptr,
                    nullptr, x.idx.data(), x.out.data(), q),
            status_t::failure);
    expect_same(x, const_cast<problem_t &>(before), 0.0f, "untouched");
}

// The batch description of gdn_direct: num_decodes in [0, batch], cu_seqlens
// from 0, non-decreasing, one token per decode and ending at num_tokens, and
// num_tokens == batch without cu_seqlens.
TEST(GdnApi, RejectsInvalidBatchDescription) {
    GDN_SKIP_UNLESS_VALIDATING();
    struct case_t {
        const char *what;
        int64_t nd;
        std::vector<int32_t> qsl; // empty: null cu_seqlens
        int64_t num_tokens;
    };
    // Three sequences over 5 rows; valid with nd = 1 and qsl {0, 1, 3, 5}.
    const case_t cases[] = {
            {"negative num_decodes", -1, {0, 1, 3, 5}, 5},
            {"num_decodes above batch", 4, {0, 1, 3, 5}, 5},
            {"cu_seqlens[0] != 0", 1, {1, 2, 3, 5}, 5},
            {"decreasing cu_seqlens", 1, {0, 1, 4, 3}, 3},
            {"decode with two tokens", 1, {0, 2, 3, 5}, 5},
            {"decode with no token", 2, {0, 1, 1, 5}, 5},
            {"num_tokens below cu_seqlens[batch]", 1, {0, 1, 3, 5}, 4},
            {"num_tokens above cu_seqlens[batch]", 1, {0, 1, 3, 4}, 5},
            {"num_tokens != batch without cu_seqlens", 1, {}, 5},
    };
    problem_t base(kSmall, 5, 4, data_type_t::f32, ng::gdn_state_layout_t::kv,
            false, 1);
    base.idx = {2, 0, 3};
    const std::vector<uint8_t> init {1, 1, 0};
    for (const case_t &c : cases) {
        SCOPED_TRACE(c.what);
        problem_t x = base;
        ng::gdn_params q = x.p;
        q.batch = 3;
        q.num_decodes = c.nd;
        q.num_tokens = c.num_tokens;
        q.cu_seqlens = c.qsl.empty() ? nullptr : c.qsl.data();
        q.has_initial_state = reinterpret_cast<const bool *>(init.data());
        for (ng::gdn_kernel_t k :
                {ng::gdn_kernel_t::reference, ng::gdn_kernel_t::none}) {
            EXPECT_EQ(x.call(q, k), status_t::failure);
        }
        expect_same(x, base, 0.0f, "untouched");
    }
    // The valid description passes.
    problem_t x = base;
    EXPECT_EQ(x.mixed(ng::gdn_kernel_t::reference, 1, {0, 1, 3, 5}, init),
            status_t::success);
}

// gdn_native_supported answers what gdn_direct's automatic selection uses:
// the CPU for default params, and the head and conv limits for a shape.
TEST(GdnApi, NativeSupportedQuery) {
    const bool isa = has_native_isa();
    EXPECT_EQ(ng::gdn_native_supported(ng::gdn_params()), isa);
    const auto shape_params = [](const shape_t &s) {
        ng::gdn_params q;
        q.num_k_heads = s.H;
        q.num_v_heads = s.HV;
        q.head_k_dim = s.K;
        q.head_v_dim = s.V;
        q.conv_width = s.W;
        return q;
    };
    for (const shape_t &s : {kQwen36, kSmall, kNoConv,
                 shape_t {1, 1, 256, 256, 8}, shape_t {4, 8, 16, 16, 1}}) {
        SCOPED_TRACE(testing::Message() << s);
        EXPECT_EQ(ng::gdn_native_supported(shape_params(s)), isa);
    }
    for (const shape_t &s : {shape_t {2, 2, 40, 32, 4},
                 shape_t {1, 1, 32, 8, 4}, shape_t {1, 1, 272, 32, 4},
                 shape_t {1, 1, 32, 272, 4}, shape_t {1, 2, 32, 32, 9}}) {
        SCOPED_TRACE(testing::Message() << s);
        EXPECT_FALSE(ng::gdn_native_supported(shape_params(s)));
    }
}

} // namespace
