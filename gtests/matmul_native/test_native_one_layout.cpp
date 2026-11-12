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

// One layout per weight: the layout the native BRGEMM keeps for a weight
// holds the bytes of the weight alone, whatever its calls carry (M, post-op,
// threads, weight cache mode, route) and in whichever order they come; and
// every flow of the native BRGEMM gives the reference algorithm's result at
// weight cache modes 0, 1 and 2, on the weight's one layout.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::matmul_config_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;
using gated_t = mm::matmul_gated_act_t;

constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kF16 = data_type_t::f16;
constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kS4 = data_type_t::s4;
constexpr data_type_t kU8 = data_type_t::u8;
constexpr int kPackNr = nat::kNativeLayoutPackNr;

/// A weight type the native BRGEMM packs, with the source of its calls and,
/// for s4, the rows of its weight scales ({groups, N}).
struct kind_t {
    const char *name;
    data_type_t wei, src;
    int groups;
};

const kind_t kKinds[] = {{"bf16", kBf16, kBf16, 0}, {"s8 x u8", kS8, kU8, 0},
        {"s8 x s8", kS8, kS8, 0}, {"s4 {1, N} x s8", kS4, kS8, 1},
        {"s4 {4, N} x s8", kS4, kS8, 4}, {"f32", kF32, kF32, 0},
        {"f16", kF16, kF16, 0}};

/// The CPU runs the native kernels of the kind.
bool served_here(const kind_t &k) {
    const nat::UarchParams &u = nat::detect_uarch();
    if (!nat::native_base_isa()) return false;
    if (k.wei == kBf16) return u.avx512bf16;
    if (k.wei == kS8 || k.wei == kS4) return u.avx512vnni;
    return true;
}

/// The post-op of a call: none, gelu (erf), a binary add of [M, N] f32, or
/// a gated activation over [gate | up] halves or (swiglu) pairs.
enum class post_t { none, gelu, add, silu, gelu_mul, gelu_tanh, swiglu };

const post_t kPosts[] = {post_t::none, post_t::gelu, post_t::add, post_t::silu,
        post_t::gelu_mul, post_t::gelu_tanh, post_t::swiglu};

const char *post_name(post_t p) {
    switch (p) {
        case post_t::gelu: return "gelu";
        case post_t::add: return "binary add";
        case post_t::silu: return "silu_and_mul";
        case post_t::gelu_mul: return "gelu_and_mul";
        case post_t::gelu_tanh: return "gelu_tanh_and_mul";
        case post_t::swiglu: return "swiglu_oai_mul";
        default: return "no post-op";
    }
}

gated_t gated_of(post_t p) {
    switch (p) {
        case post_t::silu: return gated_t::silu_and_mul;
        case post_t::gelu_mul: return gated_t::gelu_and_mul;
        case post_t::gelu_tanh: return gated_t::gelu_tanh_and_mul;
        case post_t::swiglu: return gated_t::swiglu_oai_mul;
        default: return gated_t::none;
    }
}

/// The bytes of a raw [K, N] weight of kind k (or [N, K] with transB), rows
/// tight: random values of its type (s4: random bytes, two values each).
nt::buffer_t raw_weight(
        const kind_t &k, int K, int N, bool transB, uint32_t seed) {
    if (k.wei == kS4)
        return nt::random_buffer(kU8,
                nat::native_raw_weight_bytes(kS4, K, N, transB ? K : N, transB),
                seed, 0.0f, 255.0f);
    const bool int8 = k.wei == kS8;
    return nt::random_buffer(k.wei, static_cast<size_t>(K) * N, seed,
            int8 ? -8.0f : -1.0f, int8 ? 8.0f : 1.0f);
}

/// One call on a weight of kind k at wei: its operands and its call_t.
struct op_call_t {
    nt::call_t c;
    nt::buffer_t src, dst, bin, bias, wei_scale, src_scales;
    float src_scale = 0.03125f;
    int32_t src_zp = 3;

    op_call_t(const kind_t &k, const void *wei, int M, int N, int K,
            bool transB, post_t post, uint32_t seed) {
        const bool int8 = k.wei == kS8 || k.wei == kS4;
        c.M = M;
        c.N = N;
        c.K = K;
        c.transB = transB;
        c.wei = wei;
        c.gated = gated_of(post);
        src = nt::random_buffer(k.src, static_cast<size_t>(M) * K, seed,
                k.src == kU8 ? 0.0f : (int8 ? -8.0f : -1.0f),
                int8 ? 8.0f : 1.0f);
        c.src = src.get();
        dst = nt::make_buffer(static_cast<size_t>(M) * N * 4);
        c.dst = dst.get();
        c.params.dtypes.src = k.src;
        c.params.dtypes.wei = k.wei;
        c.params.dtypes.dst = kF32;
        if (int8) {
            auto &q = c.params.quant_params;
            q.src_scale.buff = &src_scale;
            q.src_scale.dt = kF32;
            q.src_scale.dims = {1, 1};
            // An s8 source of an s4 weight takes a scale per row.
            if (k.wei == kS4) {
                src_scales = nt::random_buffer(
                        kF32, static_cast<size_t>(M), seed + 5, 0.02f, 0.04f);
                q.src_scale.buff = src_scales.get();
                q.src_scale.dims = {M, 1};
            }
            if (k.src == kU8) {
                q.src_zp.buff = &src_zp;
                q.src_zp.dt = data_type_t::s32;
                q.src_zp.dims = {1, 1};
            }
            const int rows = std::max(1, k.groups);
            wei_scale = nt::random_buffer(kF32, static_cast<size_t>(rows) * N,
                    seed + 1, 0.0625f, 0.25f);
            q.wei_scale.buff = wei_scale.get();
            q.wei_scale.dt = kF32;
            q.wei_scale.dims = {rows, N};
            if (k.wei == kS4) c.params.dtypes.compute = kS8;
        }
        if (post == post_t::gelu) {
            mm::matmul_post_op po;
            po.po_type = post_op_type_t::gelu_erf;
            c.params.postop_.push_back(po);
        }
        if (post == post_t::add) {
            bin = nt::random_buffer(
                    kF32, static_cast<size_t>(M) * N, seed + 2, -1.0f, 1.0f);
            mm::matmul_post_op po;
            po.po_type = post_op_type_t::binary_add;
            po.buff = bin.get();
            po.dtype = kF32;
            po.dims = {M, N};
            c.params.postop_.push_back(po);
        }
    }

    void set_bias(data_type_t dt, uint32_t seed) {
        bias = nt::random_buffer(
                dt, static_cast<size_t>(c.N), seed, -1.0f, 1.0f);
        c.bias = bias.get();
        c.params.dtypes.bias = dt;
    }

    /// The route: under ZENDNNL_MATMUL_ALGO=10 (pinned by the caller) or
    /// with lowoha_algo native_brgemm; weight cache mode wc and threads.
    void route(bool algo10, int32_t wc, int threads) {
        c.params.lowoha_algo
                = algo10 ? matmul_algo_t::none : matmul_algo_t::native_brgemm;
        c.params.weight_cache_type = wc;
        c.params.num_threads = threads;
    }
};

/// The raw weight request of a call, as the native BRGEMM makes it.
nat::native_raw_weight_t request_of(const nt::call_t &c) {
    nat::native_raw_weight_t w;
    w.wei_dt = c.params.dtypes.wei;
    w.weight = c.wei;
    w.K = c.K;
    w.N = c.N;
    w.ldb = c.ldb_or_tight();
    w.transB = c.transB;
    w.pack_nr = kPackNr;
    return w;
}

/// native_pack_weight of the weight raw of c's shape on threads threads.
std::vector<uint8_t> pack_of(
        const nt::call_t &c, const void *raw, int threads) {
    std::vector<uint8_t> out(
            nat::native_packed_bytes(c.params.dtypes.wei, c.K, c.N, kPackNr));
    EXPECT_EQ(nat::native_pack_weight(c.params.dtypes.wei, raw, c.K, c.N,
                      c.ldb_or_tight(), kPackNr, c.transB, out.data(), threads),
            status_t::success);
    return out;
}

/// The process-wide weight cache mode 2 while it lives, so that each call's
/// own mode applies; and the caches cleared at both ends.
struct mode_two_process_t {
    matmul_config_t &cfg = matmul_config_t::instance();
    const int32_t weight_cache = cfg.get_weight_cache();
    mode_two_process_t() {
        cfg.set_weight_cache(2);
        mm::clear_matmul_weight_caches();
    }
    ~mode_two_process_t() {
        mm::clear_matmul_weight_caches();
        cfg.set_weight_cache(weight_cache);
    }
};

/// The bytes of the one layout the native BRGEMM keeps for c's weight: the
/// buffer itself when it holds an image, else the cached layout (found, not
/// made: the weight keeps one layout).
::testing::AssertionResult kept_layout_is(
        const nt::call_t &c, const std::vector<uint8_t> &want, bool in_place) {
    if (nt::native_layouts(c.wei) != 1)
        return ::testing::AssertionFailure()
                << nt::native_layouts(c.wei) << " layouts of the weight";
    const size_t raw_bytes = nat::native_raw_weight_bytes(
            c.params.dtypes.wei, c.K, c.N, c.ldb_or_tight(), c.transB);
    if (nat::native_weight_packed_in_place(c.wei, raw_bytes) != in_place)
        return ::testing::AssertionFailure()
                << (in_place ? "no image in place" : "an image in place");
    const void *held = c.wei;
    if (!in_place) {
        nat::native_layout_t out;
        if (nat::get_native_layout(request_of(c), 1, 0, 1, out)
                        != status_t::success
                || out.owned != nullptr || out.data == nullptr)
            return ::testing::AssertionFailure() << "no cached layout";
        held = out.data;
        if (nt::native_layouts(c.wei) != 1)
            return ::testing::AssertionFailure()
                    << "the lookup made another layout";
    }
    if (std::memcmp(held, want.data(), want.size()) != 0)
        return ::testing::AssertionFailure()
                << "the layout differs from the weight's pack";
    return ::testing::AssertionSuccess();
}

/// A bf16, f16 or s4 weight of whole column blocks with tight rows, whose
/// layout has the raw weight's size (bf16: K even; s4: K a multiple of 8),
/// at weight cache mode 2 under ZENDNNL_MATMUL_ALGO=10 is rewritten in place
/// by its first call.
bool rewrites_in_place(const kind_t &k, int K, int N, int32_t wc, bool algo10) {
    const bool type = (k.wei == kBf16 && K % 2 == 0) || k.wei == kF16
            || (k.wei == kS4 && K % 8 == 0);
    return type && wc == 2 && algo10 && N % kPackNr == 0;
}

// The one layout of a weight holds the bytes native_pack_weight writes for
// the weight alone, on one thread: whatever its first call (M 1, 7, 64 or
// 512; no post-op, gelu, a binary add, silu / gelu / gelu_tanh over [gate |
// up] halves or swiglu over pairs; 1 or 8 threads; weight cache mode 1, or 2,
// which rewrites a bf16, f16 or s4 weight of whole column blocks in place
// under ZENDNNL_MATMUL_ALGO=10 and caches the others out of place; with
// lowoha_algo native_brgemm or under ZENDNNL_MATMUL_ALGO=10), and whichever
// of two calls of other M, post-ops and threads comes first; for bf16, s8
// with u8 and s8 sources, s4 with {1, N} and {4, N} weight scales, f32 and
// f16 weights, [K, N] and [N, K].
TEST(NativeOneLayout, TheLayoutOfAWeightIsTheWeightsWhateverItsCalls) {
    const mode_two_process_t process;
    const int Ms[] = {1, 7, 64, 512};
    const int K = 256, N = 128;
    uint32_t seed = 31000;
    int ran = 0;
    for (const kind_t &k : kKinds) {
        if (!served_here(k)) continue;
        for (const int32_t wc : {1, 2})
            for (int i = 0; i < 7; ++i) {
                const post_t first_post = kPosts[i],
                             next_post = kPosts[(i + 3) % 7];
                const int first_m = Ms[i % 4], next_m = Ms[(i + 2) % 4];
                const int first_threads = i % 2 ? 8 : 1;
                const bool algo10 = i % 3 != 1, transB = i == 5;
                SCOPED_TRACE(std::string(k.name) + ", mode "
                        + std::to_string(wc)
                        + (algo10 ? ", ZENDNNL_MATMUL_ALGO=10"
                                  : ", lowoha_algo native_brgemm")
                        + (transB ? ", [N, K]" : ", [K, N]") + ", first M "
                        + std::to_string(first_m) + " " + post_name(first_post)
                        + " on " + std::to_string(first_threads)
                        + " threads, then M " + std::to_string(next_m) + " "
                        + post_name(next_post));
                const nt::algo_pin_t pin(algo10 ? matmul_algo_t::native_brgemm
                                                : matmul_algo_t::none);
                const nt::buffer_t wei = nt::kept_until_exit(
                        raw_weight(k, K, N, transB, seed += 8));
                const nt::buffer_t raw = nt::make_buffer(wei.bytes);
                std::memcpy(raw.get(), wei.get(), wei.bytes);
                op_call_t first(k, wei.get(), first_m, N, K, transB, first_post,
                        seed + 1);
                first.route(algo10, wc, first_threads);
                op_call_t next(k, wei.get(), next_m, N, K, transB, next_post,
                        seed + 2);
                next.route(algo10, wc, 9 - first_threads);
                const std::vector<uint8_t> want
                        = pack_of(first.c, raw.get(), 1);
                ASSERT_EQ(nt::run_matmul(first.c), status_t::success)
                        << "the first call";
                const bool in_place = rewrites_in_place(k, K, N, wc, algo10);
                EXPECT_TRUE(kept_layout_is(first.c, want, in_place))
                        << "after the first call";
                ASSERT_EQ(nt::run_matmul(next.c), status_t::success)
                        << "the next call";
                EXPECT_TRUE(kept_layout_is(next.c, want, in_place))
                        << "after the next call";
                ++ran;
            }
    }
    if (ran == 0) GTEST_SKIP() << "no weight type of the native BRGEMM here";
}

// native_pack_weight writes the same bytes for a weight on 1, 3, 8 or 32
// threads, for every weight type the native BRGEMM packs, of whole and of
// padded column blocks, [K, N] and [N, K].
TEST(NativeOneLayout, PacksOnAnyThreadsHoldTheSameBytes) {
    if (!nat::native_base_isa()) GTEST_SKIP() << "needs AVX-512";
    uint32_t seed = 32000;
    for (const kind_t &k : kKinds)
        for (const int N : {128, 100})
            for (const bool transB : {false, true}) {
                SCOPED_TRACE(std::string(k.name) + ", N " + std::to_string(N)
                        + (transB ? ", [N, K]" : ", [K, N]"));
                const int K = 200;
                const nt::buffer_t raw = raw_weight(k, K, N, transB, seed += 4);
                op_call_t c(k, raw.get(), 1, N, K, transB, post_t::none, seed);
                const std::vector<uint8_t> one = pack_of(c.c, raw.get(), 1);
                for (const int threads : {3, 8, 32})
                    EXPECT_TRUE(pack_of(c.c, raw.get(), threads) == one)
                            << threads << " threads";
            }
}

// ── Weight cache modes 0, 1 and 2 over every flow ───────────────────────

/// One case of the matrix: a weight of kind k (K x N), its post-op, bias,
/// alpha and beta, and the M of its calls: m_a, then m_b, then m_a again.
struct flow_t {
    int K, N, m_a, m_b;
    post_t post;
    data_type_t bias;
    float alpha, beta;
};

/// The reference result of c over the raw weight raw (f16 operands widened
/// to f32, exactly): dst of c's rows and output columns, f32.
std::vector<float> reference_of(const nt::call_t &c, const kind_t &k,
        const void *raw, const void *prior) {
    nt::call_t r = c;
    nt::buffer_t wide_src, wide_wei;
    if (k.wei == kF16) {
        const size_t a = static_cast<size_t>(c.M) * c.K,
                     b = static_cast<size_t>(c.K) * c.N;
        wide_src = nt::make_buffer(a * 4);
        wide_wei = nt::make_buffer(b * 4);
        for (size_t i = 0; i < a; ++i)
            nt::store_f32(
                    wide_src.get(), kF32, i, nt::load_f32(c.src, kF16, i));
        for (size_t i = 0; i < b; ++i)
            nt::store_f32(wide_wei.get(), kF32, i, nt::load_f32(raw, kF16, i));
        r.src = wide_src.get();
        r.params.dtypes.src = kF32;
        r.params.dtypes.wei = kF32;
        raw = wide_wei.get();
    }
    const int cols = c.out_cols();
    const bool gated = c.gated != gated_t::none;
    std::vector<float> pre(static_cast<size_t>(c.M) * c.N);
    if (!gated)
        std::memcpy(pre.data(), prior, static_cast<size_t>(c.M) * cols * 4);
    EXPECT_EQ(nt::run_reference(r, raw, pre.data()), status_t::success);
    if (!gated) {
        pre.resize(static_cast<size_t>(c.M) * cols);
        return pre;
    }
    std::vector<float> out(static_cast<size_t>(c.M) * cols);
    nt::gated_act_ref(pre.data(), c.M, c.N, c.gated, out.data());
    return out;
}

/// c's dst against the reference within the band of a low-precision call
/// (the post-ops' approximations included).
::testing::AssertionResult matches_reference(const nt::call_t &c,
        const kind_t &k, const void *raw, const void *prior) {
    const std::vector<float> want = reference_of(c, k, raw, prior);
    const bool quant = k.wei == kS8 || k.wei == kS4;
    return nt::within_band(c.dst, c.out_cols(), want.data(), c.out_cols(), kF32,
            c.M, c.out_cols(), rtol_bf16,
            nt::comparison_abs_bound(
                    c.K, epsilon_bf16, c.alpha, c.beta, true, quant, false));
}

/// The cases of a kind: every post-op class at each weight cache mode, with
/// N tails (1, 15, 16, 33, 4100, 128), K tails, biases, alpha / beta and
/// both orders of M = 1 and M > 1 calls, picked in turn.
std::vector<flow_t> flows_of(const kind_t &k) {
    const bool s4 = k.wei == kS4;
    // K tails: odd, and of whole quads not octets; for s4 groups of whole
    // octets in four groups.
    const int Ks[] = {s4 ? 64 : 37, s4 ? 128 : 100, s4 ? 96 : 200};
    struct shape_t {
        int N, m_a, m_b;
    };
    const shape_t plain[] = {{1, 1, 64}, {15, 7, 1}, {16, 1, 2}, {33, 512, 1},
            {4100, 1, 7}, {128, 64, 1}, {128, 1, 512}};
    const shape_t gated[]
            = {{16, 1, 7}, {128, 512, 1}, {4100, 2, 1}, {128, 1, 64}};
    std::vector<flow_t> out;
    int i = 0;
    for (const post_t p : kPosts) {
        const bool is_gated = gated_of(p) != gated_t::none;
        for (int j = 0; j < (is_gated ? 1 : 2); ++j, ++i) {
            const shape_t s = is_gated ? gated[i % 4] : plain[i % 7];
            flow_t f;
            f.K = Ks[i % 3];
            f.N = s.N;
            // The reference of an s4 weight runs slowly, and with weight
            // scales per group row by row: those calls stop at 64 and 7 rows.
            const int rows = !s4 ? s.m_a + s.m_b : k.groups > 1 ? 7 : 64;
            f.m_a = std::min(s.m_a, rows);
            f.m_b = std::min(s.m_b, rows);
            f.post = p;
            f.bias = i % 3 == 0                      ? data_type_t::none
                    : (k.wei == kBf16 && i % 3 == 2) ? kBf16
                                                     : kF32;
            f.alpha = !is_gated && i % 4 == 1 ? 0.5f : 1.0f;
            f.beta = !is_gated && i % 4 == 2 ? 0.75f : 0.0f;
            out.push_back(f);
        }
    }
    return out;
}

/// The matrix for the kinds of wei: every case of flows_of at weight cache
/// modes 0, 1 and 2, under ZENDNNL_MATMUL_ALGO=10 (and lowoha_algo
/// native_brgemm for every other case): on a fresh weight the m_a call and
/// the m_b call give the reference's result and the m_a call again the bits
/// of the first, on one layout of the weight (an image in place where mode
/// 2 rewrites it, which the later calls read); at mode 0 a weight buffer
/// refilled with a new weight gives the new weight's result.
void run_matrix(data_type_t wei) {
    const mode_two_process_t process;
    uint32_t seed = 33000 + static_cast<uint32_t>(wei) * 1000;
    int ran = 0;
    for (const kind_t &k : kKinds) {
        if (k.wei != wei || !served_here(k)) continue;
        int n = 0;
        for (const flow_t &f : flows_of(k))
            for (const int32_t wc : {0, 1, 2}) {
                const bool algo10 = wc == 2 || ++n % 2 == 0;
                SCOPED_TRACE(std::string(k.name) + ", mode "
                        + std::to_string(wc)
                        + (algo10 ? ", ZENDNNL_MATMUL_ALGO=10"
                                  : ", lowoha_algo native_brgemm")
                        + ", K " + std::to_string(f.K) + " N "
                        + std::to_string(f.N) + ", M " + std::to_string(f.m_a)
                        + " then " + std::to_string(f.m_b) + ", "
                        + post_name(f.post)
                        + (f.bias == data_type_t::none ? ", no bias"
                                                       : ", a bias")
                        + ", alpha " + std::to_string(f.alpha) + " beta "
                        + std::to_string(f.beta));
                const nt::algo_pin_t pin(algo10 ? matmul_algo_t::native_brgemm
                                                : matmul_algo_t::none);
                const nt::buffer_t wei_buf = nt::kept_until_exit(
                        raw_weight(k, f.K, f.N, false, seed += 16));
                nt::buffer_t raw = nt::make_buffer(wei_buf.bytes);
                std::memcpy(raw.get(), wei_buf.get(), wei_buf.bytes);
                const auto one_call = [&](int M, uint32_t s, const char *which,
                                              std::vector<uint8_t> *bits) {
                    op_call_t o(
                            k, wei_buf.get(), M, f.N, f.K, false, f.post, s);
                    o.route(algo10, wc, 8);
                    if (f.bias != data_type_t::none) o.set_bias(f.bias, s + 3);
                    o.c.alpha = f.alpha;
                    o.c.beta = f.beta;
                    const nt::buffer_t prior = nt::random_buffer(kF32,
                            static_cast<size_t>(M) * o.c.out_cols(), s + 4,
                            -1.0f, 1.0f);
                    std::memcpy(o.dst.get(), prior.get(), prior.bytes);
                    const status_t st = nt::run_matmul(o.c);
                    EXPECT_EQ(st, status_t::success) << which;
                    if (st != status_t::success) return;
                    const uint8_t *got = o.dst.as<uint8_t>();
                    const size_t bytes = static_cast<size_t>(M) * o.c.out_cols()
                            * sizeof(float);
                    if (bits != nullptr && !bits->empty()) {
                        EXPECT_EQ(std::memcmp(got, bits->data(), bytes), 0)
                                << which << ": other bits than the first call";
                        return;
                    }
                    if (bits != nullptr) bits->assign(got, got + bytes);
                    EXPECT_TRUE(
                            matches_reference(o.c, k, raw.get(), prior.get()))
                            << which;
                };
                std::vector<uint8_t> first_bits;
                one_call(f.m_a, seed + 1, "the first call", &first_bits);
                one_call(f.m_b, seed + 2, "the second call", nullptr);
                one_call(f.m_a, seed + 1, "the first call again", &first_bits);
                const bool in_place
                        = rewrites_in_place(k, f.K, f.N, wc, algo10);
                EXPECT_EQ(nt::native_layouts(wei_buf.get()), wc == 0 ? 0u : 1u);
                EXPECT_EQ(nat::native_weight_packed_in_place(
                                  wei_buf.get(), wei_buf.bytes),
                        in_place);
                if (wc == 0) {
                    const nt::buffer_t next
                            = raw_weight(k, f.K, f.N, false, seed + 5);
                    std::memcpy(wei_buf.get(), next.get(), next.bytes);
                    std::memcpy(raw.get(), next.get(), next.bytes);
                    one_call(f.m_b, seed + 6, "a new weight in the buffer",
                            nullptr);
                }
                ++ran;
            }
    }
    if (ran == 0) GTEST_SKIP() << "the native BRGEMM does not serve it here";
}

TEST(NativeCacheModes, Bf16FlowsAtEveryMode) {
    run_matrix(kBf16);
}

TEST(NativeCacheModes, S8FlowsAtEveryMode) {
    run_matrix(kS8);
}

TEST(NativeCacheModes, S4FlowsAtEveryMode) {
    run_matrix(kS4);
}

TEST(NativeCacheModes, F32FlowsAtEveryMode) {
    run_matrix(kF32);
}

TEST(NativeCacheModes, F16FlowsAtEveryMode) {
    run_matrix(kF16);
}

} // namespace
