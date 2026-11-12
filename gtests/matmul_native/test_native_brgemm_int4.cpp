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

// s4 weights on the native BRGEMM (DA8W4: a bf16 / f32 source quantized in
// the call, or a static s8 source, times symmetric s4 weights with {G, N}
// weight scales): the s4 group kernel against a scalar model of its orders,
// bit for bit; matmul_direct calls under ALGO 10 against a model of the
// call's order (the fused order of a per-token or per-group source, the
// reference order of alpha / beta or an integer dst), bit for bit, against
// the reference algorithm (ALGO 9) in its order and against the default
// route's custom kernel in the fused order, bit for bit; and the calls the
// native BRGEMM does not serve.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#include <sys/mman.h>
#endif

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int4_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_call.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile_internal.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace nt = native_brgemm_test;
namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
using nt::buffer_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;
using quant_t = mm::matmul_quantization_params_t::matmul_quant_t;

constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kF16 = data_type_t::f16;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kU8 = data_type_t::u8;
constexpr data_type_t kS32 = data_type_t::s32;
constexpr data_type_t kS4 = data_type_t::s4;
constexpr data_type_t kNone = data_type_t::none;
constexpr float kSqrt2Over2 = 0.707106769084930419921875f;

const char *dt_name(data_type_t dt) {
    switch (dt) {
        case kF32: return "f32";
        case kBf16: return "bf16";
        case kF16: return "f16";
        case kS8: return "s8";
        case kU8: return "u8";
        case kS32: return "s32";
        default: return "none";
    }
}

bool vnni() {
    return nat::detect_uarch().avx512vnni;
}

TEST(NativeBrgemmInt4GroupKernel, Table) {
    for (int nv : {1, 2}) {
        const int max_mr = nat::native_int4_group_ukernel_max_mr(nv);
        EXPECT_EQ(max_mr, nv == 1 ? 12 : 6) << nv;
        for (const auto order : {nat::native_int8_order_t::fused,
                     nat::native_int8_order_t::reference}) {
            EXPECT_EQ(nat::select_native_int4_group_ukernel(0, nv, order),
                    nullptr);
            EXPECT_EQ(nat::select_native_int4_group_ukernel(
                              max_mr + 1, nv, order),
                    nullptr);
            for (int mr = 1; mr <= max_mr; ++mr)
                EXPECT_NE(nat::select_native_int4_group_ukernel(mr, nv, order),
                        nullptr)
                        << mr << " " << nv;
        }
        for (int mr = 1; mr <= max_mr; ++mr) {
            const auto fused = nat::select_native_int4_group_ukernel(
                    mr, nv, nat::native_int8_order_t::fused);
            const auto groups = nat::select_native_int4_group_ukernel(
                    mr, nv, nat::native_int8_order_t::fused, true);
            const auto reference = nat::select_native_int4_group_ukernel(
                    mr, nv, nat::native_int8_order_t::reference);
            EXPECT_NE(groups, nullptr) << mr << " " << nv;
            EXPECT_NE(fused, reference) << mr << " " << nv;
            EXPECT_NE(groups, fused) << mr << " " << nv;
            EXPECT_NE(groups, reference) << mr << " " << nv;
            EXPECT_EQ(nat::select_native_int4_group_ukernel(mr, nv,
                              nat::native_int8_order_t::reference, true),
                    reference)
                    << mr << " " << nv;
        }
    }
    for (int nv : {3, 4}) {
        EXPECT_EQ(nat::native_int4_group_ukernel_max_mr(nv), 0);
        EXPECT_EQ(nat::select_native_int4_group_ukernel(
                          1, nv, nat::native_int8_order_t::fused),
                nullptr);
    }
}

// The batch-reduce contract of the s4 group kernel on its own: per group
// the exact int32 sum of its shift and the s8 source times the weight read
// as w + 8 (the low nibbles of an octet one K quad, the high nibbles the
// next), groups of an odd number of quads starting or ending half way
// through an octet; then, in the fused order, one FMA with the weight scale
// per group and the source scale after the last batch, or one FMA with the
// product of the group's source and weight scales (zero where the group's
// sum is zero), or, in the reference order, source and weight scales and
// one add per group (f32, bf16 or f16 weight scales); from C (or +0.0),
// stored to C or to a bf16 / f32 destination.  Bit for bit against a scalar
// model over every row count, width, order and scale type.  Nothing past
// the tile's columns is written, and C is not written when the tile stores
// to a destination.
TEST(NativeBrgemmInt4GroupKernel, MatchesTheModel) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    enum class store_t { c, f32, bf16 };
    std::mt19937 rng(20261106);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_int_distribution<int> s8(-128, 127);
    std::uniform_real_distribution<float> scale(0.002f, 0.01f);
    std::mt19937 draws(11);
    std::uniform_int_distribution<int> pick(0, 71);
    int variant = 0;
    struct kind_t {
        nat::native_int8_order_t order;
        bool src_groups;
    };
    for (const kind_t kind : {kind_t {nat::native_int8_order_t::fused, false},
                 kind_t {nat::native_int8_order_t::fused, true},
                 kind_t {nat::native_int8_order_t::reference, false}})
        for (const data_type_t sb_dt : {kF32, kBf16, kF16})
            for (int nv : {1, 2})
                for (int mr = 1;
                        mr <= nat::native_int4_group_ukernel_max_mr(nv); ++mr)
                    for (int k_quads : {1, 2, 3, 5, 8, 32})
                        for (int batch : {1, 2, 5}) {
                            const int v = variant++;
                            const auto order = kind.order;
                            const bool fused
                                    = order == nat::native_int8_order_t::fused;
                            const bool src_groups = kind.src_groups;
                            const int nr = 16 * nv;
                            const int pack_nr = 64, col0 = nv == 1 ? 16 : 32;
                            const int s = pick(draws);
                            const store_t store = static_cast<store_t>(s % 3);
                            const bool accumulate = (s / 3) % 2 == 1;
                            const bool finish
                                    = fused && !src_groups && (s / 6) % 2 == 0;
                            const bool per_group_src = src_groups
                                    || (!fused && (s / 12) % 2 == 0);
                            const int lda = batch * k_quads * 4 + 4 * (s % 4);
                            const int quad0 = (s / 24) % 2;
                            // A zero sum with a negative scale gives -0.0,
                            // which the first group adds to +0.0.
                            const bool zero_row = v % 5 == 2;
                            SCOPED_TRACE(std::string("order=")
                                    + (fused ? "fused" : "reference")
                                    + (src_groups ? " (source groups)" : "")
                                    + " mr=" + std::to_string(mr)
                                    + " nv=" + std::to_string(nv) + " k_quads="
                                    + std::to_string(k_quads) + " batch="
                                    + std::to_string(batch) + " quad0="
                                    + std::to_string(quad0) + " store="
                                    + std::to_string(static_cast<int>(store))
                                    + " accumulate="
                                    + std::to_string(accumulate) + " finish="
                                    + std::to_string(finish) + " per_group_src="
                                    + std::to_string(per_group_src)
                                    + " lda=" + std::to_string(lda)
                                    + " wei_scale=" + dt_name(sb_dt));
                            std::vector<int8_t> a(
                                    static_cast<size_t>(mr) * lda);
                            for (auto &x : a)
                                x = static_cast<int8_t>(s8(rng));
                            if (zero_row)
                                std::fill(
                                        a.begin(), a.begin() + lda, int8_t {0});
                            const int quads = quad0 + batch * k_quads;
                            const int octets = (quads + 1) / 2;
                            const int b_stride = pack_nr * 4;
                            std::vector<uint8_t> b(
                                    static_cast<size_t>(octets) * b_stride);
                            for (auto &x : b)
                                x = static_cast<uint8_t>(byte(rng));
                            // The weight at the batch's K value k (from quad
                            // quad0 on) and tile column n, w = nibble - 8.
                            const auto w_at = [&](int k, int n) {
                                const int q = quad0 + k / 4;
                                const uint8_t x = b[static_cast<size_t>(q / 2)
                                                * b_stride
                                        + static_cast<size_t>(col0 + n) * 4
                                        + k % 4];
                                return (q % 2 ? x >> 4 : x & 0x0F) - 8;
                            };
                            std::vector<int32_t> shift(
                                    static_cast<size_t>(mr) * batch);
                            for (int m = 0; m < mr; ++m)
                                for (int g = 0; g < batch; ++g) {
                                    int32_t sum = 0;
                                    for (int k = 0; k < k_quads * 4; ++k)
                                        sum += a[static_cast<size_t>(m) * lda
                                                + static_cast<size_t>(g)
                                                        * k_quads * 4
                                                + k];
                                    shift[static_cast<size_t>(m) * batch + g]
                                            = -8 * sum;
                                }
                            const int sa_ld = per_group_src ? batch : 1;
                            const int sa_step = per_group_src ? 1 : 0;
                            std::vector<float> sa(
                                    static_cast<size_t>(mr) * batch);
                            for (auto &x : sa)
                                x = scale(rng) * (zero_row ? -1.0f : 1.0f);
                            const int sb_step = nr + 16;
                            const size_t sb_n
                                    = static_cast<size_t>(batch) * sb_step;
                            buffer_t sb = nt::make_buffer(
                                    sb_n * nt::dt_bytes(sb_dt));
                            for (size_t i = 0; i < sb_n; ++i)
                                nt::store_f32(sb.get(), sb_dt, i,
                                        scale(rng) * (i % 7 == 3 ? -1 : 1));
                            const int ldc = nr + 16, dst_ldc = nr + 32;
                            std::vector<float> c(static_cast<size_t>(mr) * ldc);
                            for (auto &x : c)
                                x = scale(rng) * 100.0f - 0.5f;
                            const std::vector<float> c_in = c;
                            std::vector<uint8_t> dst(
                                    static_cast<size_t>(mr) * dst_ldc * 4,
                                    0x5A);
                            const std::vector<uint8_t> dst_in = dst;

                            nat::native_int4_group_ukernel_args_t args;
                            args.A = reinterpret_cast<const uint8_t *>(
                                    a.data());
                            args.lda = lda;
                            args.a_step = static_cast<size_t>(k_quads) * 4;
                            args.B = b.data() + col0 * 4;
                            args.b_stride = b_stride;
                            args.quad0 = quad0;
                            args.k_quads = k_quads;
                            args.batch = batch;
                            args.shift = shift.data();
                            args.shift_ld = batch;
                            args.shift_step = 1;
                            args.src_scale = sa.data();
                            args.src_scale_ld = sa_ld;
                            args.src_scale_step = sa_step;
                            args.wei_scale = sb.get();
                            args.wei_scale_step = sb_step;
                            args.wei_scale_dt = sb_dt;
                            args.C = c.data();
                            args.ldc = ldc;
                            args.accumulate = accumulate;
                            args.finish = finish;
                            if (store != store_t::c) {
                                args.dst = dst.data();
                                args.dst_ldc = dst_ldc;
                                args.dst_bf16 = store == store_t::bf16;
                            }
                            nat::select_native_int4_group_ukernel(
                                    mr, nv, order, src_groups)(args);

                            size_t bad = 0;
                            for (int m = 0; m < mr; ++m)
                                for (int n = 0; n < nr; ++n) {
                                    float sum = accumulate
                                            ? c_in[static_cast<size_t>(m) * ldc
                                                      + n]
                                            : 0.0f;
                                    for (int g = 0; g < batch; ++g) {
                                        int64_t exact = 0;
                                        for (int k = 0; k < k_quads * 4; ++k) {
                                            const int kk = g * k_quads * 4 + k;
                                            exact += static_cast<int64_t>(
                                                             w_at(kk, n))
                                                    * a[static_cast<size_t>(m)
                                                                    * lda
                                                            + kk];
                                        }
                                        const float d
                                                = static_cast<float>(exact);
                                        const float ws = nt::load_f32(sb.get(),
                                                sb_dt,
                                                static_cast<size_t>(g) * sb_step
                                                        + n);
                                        const float sa_g
                                                = sa[static_cast<size_t>(m)
                                                                * sa_ld
                                                        + static_cast<size_t>(g)
                                                                * sa_step];
                                        if (src_groups) {
                                            const float sc = exact != 0
                                                    ? sa_g * ws
                                                    : 0.0f;
                                            sum = std::fma(d, sc, sum);
                                        } else if (fused) {
                                            sum = std::fma(d, ws, sum);
                                        } else {
                                            float t = d * sa_g;
                                            t = t * ws;
                                            sum = sum + t;
                                        }
                                    }
                                    if (finish)
                                        sum = sum
                                                * sa[static_cast<size_t>(m)
                                                        * sa_ld];
                                    const size_t ci
                                            = static_cast<size_t>(m) * ldc + n;
                                    const size_t di
                                            = static_cast<size_t>(m) * dst_ldc
                                            + n;
                                    if (store == store_t::c) {
                                        if (std::memcmp(&c[ci], &sum, 4) != 0)
                                            ++bad;
                                    } else if (store == store_t::f32) {
                                        if (std::memcmp(dst.data() + 4 * di,
                                                    &sum, 4)
                                                != 0)
                                            ++bad;
                                    } else {
                                        uint16_t want = 0;
                                        nt::store_f32(&want, kBf16, 0, sum);
                                        if (std::memcmp(dst.data() + 2 * di,
                                                    &want, 2)
                                                != 0)
                                            ++bad;
                                    }
                                }
                            EXPECT_EQ(bad, 0u);
                            const size_t esz = store == store_t::bf16 ? 2 : 4;
                            size_t written = 0;
                            for (int m = 0; m < mr; ++m)
                                for (int n = 0; n < ldc; ++n) {
                                    const size_t ci
                                            = static_cast<size_t>(m) * ldc + n;
                                    if (store == store_t::c && n < nr) continue;
                                    written += std::memcmp(&c[ci], &c_in[ci], 4)
                                            != 0;
                                }
                            for (size_t i = 0; i < dst.size(); ++i) {
                                const size_t row = i / (dst_ldc * esz);
                                const size_t col = i % (dst_ldc * esz) / esz;
                                if (store != store_t::c
                                        && row < static_cast<size_t>(mr)
                                        && col < static_cast<size_t>(nr))
                                    continue;
                                written += dst[i] != dst_in[i];
                            }
                            EXPECT_EQ(written, 0u);
                        }
}

// ── DA8W4 calls ─────────────────────────────────────────────────────────

/// A bf16 / f32 source quantized in the call to s8, or a static s8 source.
enum class src_t { bf16, f32, s8 };
/// Source scale: {1, 1}, {M, 1} or {M, G}.
enum class gran_t { tensor, token, group };
/// Post-ops: binary add of an [M, N] f32 matrix, or gelu (erf).
enum class po_t { none, add, gelu };

struct case_t {
    std::string name;
    int M = 16, N = 64, gs = 32, G = 4; ///< K = gs * G
    src_t src = src_t::bf16;
    gran_t gran = gran_t::token;
    bool transA = false, transB = true;
    int ldb_pad = 0; ///< elements past the row length of the weight
    data_type_t dst = kBf16, bias = kNone;
    data_type_t wei_scale_dt = kBf16, src_scale_dt = kBf16;
    float alpha = 1.0f, beta = 0.0f;
    po_t po = po_t::none;
    mm::matmul_gated_act_t gated = mm::matmul_gated_act_t::none;
    int wc = 1; ///< weight cache mode of the call
    int threads = 4; ///< 0: every thread
    int calls = 1; ///< native calls on the same weight
    /// The call goes to native_brgemm_execute directly: matmul_direct does
    /// not take it (an integer dst, an f32 source with a bf16 dst).
    bool direct = false;
    /// The full tiles of the call over its plain pack take their groups in
    /// more than one batch, and with odd_batches the batches past the first
    /// start on the high nibbles of a K octet.
    bool batches = false, odd_batches = false;
    /// The full tiles of the call over its plain pack take its one group in
    /// more than one K pass.
    bool passes = false;
    int K() const { return gs * G; }
    bool integer_dst() const { return dst == kS8 || dst == kU8 || dst == kS32; }
    /// The order decide_int8_call takes for the call.
    bool fused() const {
        return alpha == 1.0f && beta == 0.0f && (dst == kF32 || dst == kBf16);
    }
    /// The source scales are per group of the weight's groups.
    bool src_groups() const { return gran == gran_t::group && G > 1; }
    /// The reference order over one group over K: alpha before the source
    /// and weight scales, beta after them into an f32 or bf16 dst and before
    /// them into an integer one.
    bool scales_after_group() const {
        return G == 1 && (alpha != 1.0f || beta != 0.0f);
    }
};

void PrintTo(const case_t &c, std::ostream *os) {
    *os << c.name;
}

std::string describe(const case_t &c) {
    static const char *const kSrc[] = {"dynbf16", "dynf32", "s8"};
    static const char *const kGran[] = {"tensor", "token", "group"};
    static const char *const kPo[] = {"", "_add", "_gelu"};
    std::string s = std::string(kSrc[static_cast<int>(c.src)]) + "_"
            + kGran[static_cast<int>(c.gran)] + "_m" + std::to_string(c.M)
            + "_n" + std::to_string(c.N) + "_gs" + std::to_string(c.gs) + "x"
            + std::to_string(c.G) + "_" + dt_name(c.dst) + "_ws"
            + dt_name(c.wei_scale_dt) + "_ss" + dt_name(c.src_scale_dt);
    if (c.transA) s += "_ta";
    if (!c.transB) s += "_kn";
    if (c.ldb_pad != 0) s += "_ldb" + std::to_string(c.ldb_pad);
    if (c.bias != kNone) s += std::string("_bias") + dt_name(c.bias);
    s += kPo[static_cast<int>(c.po)];
    if (c.gated != mm::matmul_gated_act_t::none)
        s += "_gated" + std::to_string(static_cast<int>(c.gated));
    if (c.alpha != 1.0f || c.beta != 0.0f)
        s += "_a" + std::to_string(static_cast<int>(c.alpha * 100)) + "b"
                + std::to_string(static_cast<int>(c.beta * 100));
    s += "_wc" + std::to_string(c.wc) + "_t" + std::to_string(c.threads);
    if (c.calls != 1) s += "_x" + std::to_string(c.calls);
    if (c.direct) s += "_direct";
    for (char &ch : s)
        if (ch == '-') ch = 'm';
    return s;
}

uint32_t seed_of(const std::string &s) {
    uint32_t h = 2166136261u;
    for (char ch : s)
        h = (h ^ static_cast<uint8_t>(ch)) * 16777619u;
    return h;
}

void set_quant(quant_t &q, const void *buff, data_type_t dt,
        std::vector<int64_t> dims) {
    q.buff = buff;
    q.dt = dt;
    q.dims = std::move(dims);
}

buffer_t copy_of(const buffer_t &b) {
    buffer_t c = nt::make_buffer(b.bytes);
    std::memcpy(c.get(), b.get(), b.bytes);
    return c;
}

int64_t src_scale_count(const case_t &c) {
    return c.gran == gran_t::tensor   ? 1
            : c.gran == gran_t::token ? c.M
                                      : static_cast<int64_t>(c.M) * c.G;
}

struct operands_t {
    int lda = 0, ldb = 0, ldc = 0;
    buffer_t src, wei, bias, wei_scale, src_scale, bin, init;
    mm::matmul_params params;
};

operands_t make_operands(const case_t &c) {
    operands_t o;
    const uint32_t seed = seed_of(c.name);
    const int M = c.M, N = c.N, K = c.K();
    const bool gated = c.gated != mm::matmul_gated_act_t::none;
    o.lda = c.transA ? M : K;
    o.ldb = (c.transB ? K : N) + c.ldb_pad;
    o.ldc = gated ? N / 2 : N;
    const size_t a_n = static_cast<size_t>(M) * K;
    const data_type_t src_dt = c.src == src_t::s8 ? kS8
            : c.src == src_t::bf16                ? kBf16
                                                  : kF32;
    o.src = src_dt == kS8 ? nt::random_buffer(kS8, a_n, seed, -128, 127)
                          : nt::random_buffer(src_dt, a_n, seed, -1.0f, 1.0f);
    o.wei = nt::kept_until_exit(nt::random_buffer(kU8,
            nat::native_raw_weight_bytes(kS4, K, N, o.ldb, c.transB), seed + 1,
            0, 255));
    o.wei_scale = nt::random_buffer(c.wei_scale_dt,
            static_cast<size_t>(c.G) * N, seed + 2, 0.002f, 0.01f);
    mm::matmul_params &p = o.params;
    p.dtypes.src = src_dt;
    p.dtypes.wei = kS4;
    p.dtypes.dst = c.dst;
    p.dtypes.bias = c.bias;
    p.dtypes.compute = kS8;
    p.num_threads = c.threads;
    p.weight_cache_type = c.wc;
    auto &qp = p.quant_params;
    set_quant(qp.wei_scale, o.wei_scale.get(), c.wei_scale_dt, {c.G, N});
    const int64_t n_scale = src_scale_count(c);
    if (c.src == src_t::s8) {
        o.src_scale = nt::random_buffer(
                c.src_scale_dt, n_scale, seed + 3, 0.002f, 0.01f);
    } else {
        p.dynamic_quant = true;
        o.src_scale = nt::make_buffer(n_scale * nt::dt_bytes(c.src_scale_dt));
    }
    set_quant(qp.src_scale, o.src_scale.get(), c.src_scale_dt,
            c.gran == gran_t::tensor          ? std::vector<int64_t> {1, 1}
                    : c.gran == gran_t::token ? std::vector<int64_t> {M, 1}
                                              : std::vector<int64_t> {M, c.G});
    if (c.bias != kNone)
        o.bias = nt::random_buffer(c.bias, N, seed + 4, -1.0f, 1.0f);
    if (c.po == po_t::gelu) {
        mm::matmul_post_op po;
        po.po_type = post_op_type_t::gelu_erf;
        p.postop_.push_back(po);
    }
    if (c.po == po_t::add) {
        o.bin = nt::random_buffer(
                kF32, static_cast<size_t>(M) * N, seed + 5, -1.0f, 1.0f);
        mm::matmul_post_op po;
        po.po_type = post_op_type_t::binary_add;
        po.buff = o.bin.get();
        po.dtype = kF32;
        po.dims = {M, N};
        p.postop_.push_back(po);
    }
    const float init_lo = !c.integer_dst() ? -2.0f
            : c.dst == kU8                 ? 0.0f
                                           : -20.0f;
    o.init = nt::random_buffer(c.dst, static_cast<size_t>(M) * o.ldc, seed + 6,
            init_lo, init_lo + (c.integer_dst() ? 40.0f : 4.0f));
    return o;
}

/// The call on weight wei into dst; a source scale buffer per call, since a
/// dynamic quantization writes its scales there.
nt::call_t make_call(const case_t &c, const operands_t &o, const void *wei,
        void *dst, const buffer_t &src_scale) {
    nt::call_t call;
    call.M = c.M;
    call.N = c.N;
    call.K = c.K();
    call.transA = c.transA;
    call.transB = c.transB;
    call.lda = o.lda;
    call.ldb = o.ldb;
    call.ldc = o.ldc;
    call.alpha = c.alpha;
    call.beta = c.beta;
    call.src = o.src.get();
    call.wei = wei;
    call.bias = c.bias == kNone ? nullptr : o.bias.get();
    call.dst = dst;
    call.gated = c.gated;
    call.params = o.params;
    call.params.quant_params.src_scale.buff = src_scale.get();
    return call;
}

/// The s8 source rows [M][K] and their scales [M][G] the native route
/// multiplies: a static source as given, a dynamic one quantized here by the
/// shared quantizer as the route does, a transposed one in its own layout.
struct source_t {
    std::vector<int32_t> a;
    std::vector<float> scale;
};

source_t source_of(const case_t &c, const operands_t &o) {
    const int M = c.M, K = c.K(), G = c.G;
    source_t s;
    s.a.resize(static_cast<size_t>(M) * K);
    s.scale.resize(static_cast<size_t>(M) * G);
    const auto at = [&](int m, int k) {
        return c.transA ? static_cast<size_t>(k) * o.lda + m
                        : static_cast<size_t>(m) * o.lda + k;
    };
    const void *scales = o.src_scale.get();
    buffer_t written;
    if (c.src == src_t::s8) {
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                s.a[static_cast<size_t>(m) * K + k]
                        = o.src.as<int8_t>()[at(m, k)];
    } else {
        const data_type_t dt = o.params.dtypes.src;
        const size_t es = nt::dt_bytes(dt);
        written = copy_of(o.src_scale);
        mm::matmul_params q = o.params;
        q.quant_params.src_scale.buff = written.get();
        const void *src = o.src.get();
        int quantized_lda = o.lda;
        size_t elem_size = es;
        mm::matmul_batch_params_t batch;
        mm::reorder_quant_buffers_t buffers;
        EXPECT_EQ(mm::reorder_quantization_wrapper(src, o.lda, quantized_lda,
                          elem_size, q, batch, c.transA, M, K, 1, buffers),
                status_t::success);
        EXPECT_EQ(q.dtypes.src, kS8);
        const auto *q8 = static_cast<const int8_t *>(src);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k) {
                const size_t i = c.transA
                        ? static_cast<size_t>(k) * quantized_lda + m
                        : static_cast<size_t>(m) * quantized_lda + k;
                s.a[static_cast<size_t>(m) * K + k] = q8[i];
            }
        scales = written.get();
    }
    for (int m = 0; m < M; ++m)
        for (int g = 0; g < G; ++g) {
            const size_t i = c.gran == gran_t::tensor ? 0
                    : c.gran == gran_t::token         ? static_cast<size_t>(m)
                                              : static_cast<size_t>(m) * G + g;
            s.scale[static_cast<size_t>(m) * G + g]
                    = nt::load_f32(scales, c.src_scale_dt, i);
        }
    return s;
}

/// The s4 value of weight element (k, n): two's complement, two per byte at
/// the element's linear index.
int weight_at(const case_t &c, const operands_t &o, int k, int n) {
    const size_t i = c.transB ? static_cast<size_t>(n) * o.ldb + k
                              : static_cast<size_t>(k) * o.ldb + n;
    const int v = (o.wei.as<uint8_t>()[i / 2] >> (i % 2 * 4)) & 0x0F;
    return v >= 8 ? v - 16 : v;
}

/// Pre-activation element (m, n) of the call before its store: the groups
/// in the call's order (one FMA per group and the source scale after, one
/// FMA per group with the product of its source and weight scales, zero
/// where the group's sum is zero, or one rounding per operation), then
/// alpha, beta, the bias and the post-ops, as the reference algorithm
/// orders them: over one group over K, alpha comes before the source and
/// weight scales, and beta after them into an f32 or bf16 dst, before them
/// into any other.
float model_at(
        const case_t &c, const operands_t &o, const source_t &s, int m, int n) {
    const int K = c.K(), G = c.G, gs = c.gs;
    const bool fused = c.fused();
    const bool after = c.scales_after_group();
    const bool beta_last = c.dst == kF32 || c.dst == kBf16;
    // beta of a gated call reads the pre-activation columns: a call of
    // beta is not gated here.
    const auto beta_term = [&]() {
        return nt::load_f32(
                       o.init.get(), c.dst, static_cast<size_t>(m) * o.ldc + n)
                * c.beta;
    };
    const auto add_alpha_beta = [&](float v) {
        if (c.alpha != 1.0f) v *= c.alpha;
        if (c.beta != 0.0f) v += beta_term();
        return v;
    };
    float sum = 0.0f;
    for (int g = 0; g < G; ++g) {
        int64_t d = 0;
        for (int k = g * gs; k < (g + 1) * gs; ++k)
            d += static_cast<int64_t>(s.a[static_cast<size_t>(m) * K + k])
                    * weight_at(c, o, k, n);
        const float ws = nt::load_f32(o.wei_scale.get(), c.wei_scale_dt,
                static_cast<size_t>(g) * c.N + n);
        const float sa = s.scale[static_cast<size_t>(m) * G + g];
        if (fused && c.src_groups()) {
            sum = std::fma(static_cast<float>(d), d != 0 ? sa * ws : 0.0f, sum);
        } else if (fused) {
            sum = std::fma(static_cast<float>(d), ws, sum);
        } else {
            float v = static_cast<float>(d);
            if (after && c.alpha != 1.0f) v *= c.alpha;
            if (after && c.beta != 0.0f && !beta_last) v += beta_term();
            v = v * sa;
            v = v * ws;
            if (after && c.beta != 0.0f && beta_last) v += beta_term();
            sum = after ? v : sum + v;
        }
    }
    if (fused && !c.src_groups())
        sum = sum * s.scale[static_cast<size_t>(m) * G];
    if (!after) sum = add_alpha_beta(sum);
    if (c.bias != kNone) sum += nt::load_f32(o.bias.get(), c.bias, n);
    if (c.po == po_t::gelu)
        sum = 0.5f * sum * (1.0f + std::erf(sum * kSqrt2Over2));
    if (c.po == po_t::add)
        sum = sum + o.bin.as<float>()[static_cast<size_t>(m) * c.N + n];
    return sum;
}

/// Element i of a dst of type dt as the reference algorithm stores v: an
/// integer dst clamped to its range and rounded to nearest even.
void store_ref(void *out, data_type_t dt, size_t i, float v) {
    if (dt != kS8 && dt != kU8 && dt != kS32) {
        nt::store_f32(out, dt, i, v);
        return;
    }
    const float lo = dt == kU8 ? 0.0f : dt == kS8 ? -128.0f : -2147483648.0f;
    const float hi = dt == kU8 ? 255.0f : dt == kS8 ? 127.0f : 2147483520.0f;
    nt::store_f32(out, dt, i, std::nearbyint(std::min(std::max(v, lo), hi)));
}

/// The values the call stores, in dst order (ldc apart): the gated
/// activation over the model's pre-activation, or the model.
buffer_t model_dst(const case_t &c, const operands_t &o, const source_t &s) {
    buffer_t out = copy_of(o.init);
    if (c.gated == mm::matmul_gated_act_t::none) {
        for (int m = 0; m < c.M; ++m)
            for (int n = 0; n < c.N; ++n)
                store_ref(out.get(), c.dst, static_cast<size_t>(m) * o.ldc + n,
                        model_at(c, o, s, m, n));
        return out;
    }
    std::vector<float> pre(static_cast<size_t>(c.M) * c.N);
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n)
            pre[static_cast<size_t>(m) * c.N + n] = model_at(c, o, s, m, n);
    std::vector<float> act(static_cast<size_t>(c.M) * (c.N / 2));
    nt::gated_act_ref(pre.data(), c.M, c.N, c.gated, act.data());
    for (size_t i = 0; i < act.size(); ++i)
        store_ref(out.get(), c.dst, i, act[i]);
    return out;
}

/// The model's values within the band of the epilogue's transcendental
/// functions (a gelu post-op or a gated activation); bf16 adds half an ulp.
bool transcendental(const case_t &c) {
    return c.po == po_t::gelu || c.gated != mm::matmul_gated_act_t::none;
}

::testing::AssertionResult near_model(
        const case_t &c, int ldc, const void *got, const void *want) {
    const int cols = c.gated != mm::matmul_gated_act_t::none ? c.N / 2 : c.N;
    if (c.integer_dst())
        return nt::within_steps(got, ldc, want, ldc, c.dst, c.M, cols, 1);
    return nt::within_band(got, ldc, want, ldc, c.dst, c.M, cols,
            c.dst == kBf16 ? 3e-4f + 1.0f / 256 : 3e-4f, 3e-4f);
}

/// The call through native_brgemm_execute on its raw weight, packed for the
/// call only: matmul_direct's checks of the call do not run.
status_t run_direct(const nt::call_t &c) {
    nat::GemmDescriptor desc;
    desc.M = c.M;
    desc.N = c.N;
    desc.K = c.K;
    desc.lda = c.lda_or_tight();
    desc.ldb = c.ldb_or_tight();
    desc.ldc = c.ldc_or_tight();
    desc.transA = c.transA;
    desc.transB = c.transB;
    desc.alpha = c.alpha;
    desc.beta = c.beta;
    desc.src_dt = c.params.dtypes.src;
    desc.wei_dt = c.params.dtypes.wei;
    desc.dst_dt = c.params.dtypes.dst;
    desc.bias_dt = c.params.dtypes.bias;
    desc.bias = c.bias;
    desc.is_weights_const = true;
    desc.num_threads = c.params.num_threads;
    desc.gated_act = c.gated;
    nat::native_weight_ref_t ref;
    ref.weight = c.wei;
    ref.pack_nr = nat::kNativeLayoutPackNr;
    ref.weight_cache_mode = 0;
    return nat::native_brgemm_execute(
            desc, ref, c.src, c.dst, c.bias, c.params);
}

/// The problem matmul_direct plans for the call.
nat::native_problem_t problem_of(const nt::call_t &c) {
    nat::GemmDescriptor desc;
    desc.M = c.M;
    desc.N = c.N;
    desc.K = c.K;
    desc.lda = c.lda_or_tight();
    desc.ldb = c.ldb_or_tight();
    desc.ldc = c.ldc_or_tight();
    desc.transA = c.transA;
    desc.transB = c.transB;
    desc.alpha = c.alpha;
    desc.beta = c.beta;
    desc.src_dt = c.params.dtypes.src;
    desc.wei_dt = c.params.dtypes.wei;
    desc.dst_dt = c.params.dtypes.dst;
    desc.bias_dt = c.params.dtypes.bias;
    desc.bias = c.bias;
    desc.num_threads = c.params.num_threads;
    desc.gated_act = c.gated;
    return nat::make_native_problem(desc, c.params, c.bias,
            nat::kNativeLayoutPackNr,
            nat::decide_int8_call(desc, c.params, c.bias));
}

/// The batches of groups the full tiles of the call over its plain pack take
/// at the process planner's plan, at each of their call widths, as
/// compute_int4_group_tile cuts them (native_int8_group_span): more than
/// one, and with odd the batches past the first starting on the high
/// nibbles of a K octet (an odd number of K quads per batch).
::testing::AssertionResult tiles_take_batches(
        const case_t &c, const operands_t &o, bool odd) {
    const buffer_t scale = copy_of(o.src_scale);
    const nat::native_plan_t plan = nat::native_planner().plan(
            problem_of(make_call(c, o, o.wei.get(), nullptr, scale)));
    const int rows = std::min(plan.mb, c.M);
    const int quads = c.gs / 4;
    for (int col = 0; col < plan.nr;) {
        const int cw = nat::native_int8_col_width(plan, plan.nr - col);
        const int max_mr = nat::native_int8_call_rows(plan, cw / 16);
        const int span = nat::native_int8_group_span(
                plan, c.G, quads, cw, rows, max_mr, 2);
        if (span >= c.G)
            return ::testing::AssertionFailure()
                    << "one batch of " << c.G << " groups at " << cw
                    << " columns";
        if (odd && span * quads % 2 == 0)
            return ::testing::AssertionFailure()
                    << "batches of " << span << " groups of " << quads
                    << " quads start on whole octets";
        col += cw;
    }
    return ::testing::AssertionSuccess();
}

/// The K passes of the one group the full tiles of the call over its plain
/// pack take at the process planner's plan, at each of their call widths,
/// as compute_int4_group_tile cuts them (native_int4_pass_quads): more than
/// one.
::testing::AssertionResult tiles_take_passes(
        const case_t &c, const operands_t &o) {
    const buffer_t scale = copy_of(o.src_scale);
    const nat::native_plan_t plan = nat::native_planner().plan(
            problem_of(make_call(c, o, o.wei.get(), nullptr, scale)));
    const int rows = std::min(plan.mb, c.M);
    const int quads = c.K() / 4;
    for (int col = 0; col < plan.nr;) {
        const int cw = nat::native_int8_col_width(plan, plan.nr - col);
        const int max_mr = nat::native_int8_call_rows(plan, cw / 16);
        const int pass
                = nat::native_int4_pass_quads(plan, quads, cw, rows, max_mr);
        if (pass >= quads)
            return ::testing::AssertionFailure()
                    << "one pass of " << quads << " quads at " << cw
                    << " columns";
        col += cw;
    }
    return ::testing::AssertionSuccess();
}

/// The call on the native BRGEMM into a copy of the initial dst, calls times
/// (through matmul_direct under ALGO 10 unless direct); every call must be
/// served, each with the same result.
buffer_t run_native(const case_t &c, const operands_t &o) {
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    buffer_t dst, first;
    for (int i = 0; i < c.calls; ++i) {
        dst = copy_of(o.init);
        const buffer_t scale = copy_of(o.src_scale);
        const nt::call_t call = make_call(c, o, o.wei.get(), dst.get(), scale);
        EXPECT_EQ(c.direct ? run_direct(call) : nt::run_matmul(call),
                status_t::success)
                << "call " << i;
        if (!c.direct && c.wc != 0) {
            EXPECT_GE(nt::native_layouts(o.wei.get()), 1u)
                    << "call " << i << " was not served by the native BRGEMM";
        }
        if (i == 0) {
            first = dst;
            continue;
        }
        EXPECT_TRUE(nt::within_steps(
                dst.get(), o.ldc, first.get(), o.ldc, c.dst, c.M, o.ldc, 0))
                << "call " << i << " vs call 0";
    }
    return dst;
}

/// The call rewrites its weight in place: weight cache 2 under the
/// process-wide ALGO 10 (run_native pins it), on a contiguous weight whose
/// layout is its own size (K a multiple of 8, N of 32).
bool in_place(const case_t &c) {
    return !c.direct && c.wc == 2 && c.ldb_pad == 0 && c.N % 32 == 0
            && c.K() % 8 == 0;
}

/// The native layout of the raw s4 weight raw of the call.
std::vector<uint8_t> layout_of(
        const case_t &c, const operands_t &o, const buffer_t &raw) {
    std::vector<uint8_t> out(nat::native_packed_bytes(
            kS4, c.K(), c.N, nat::kNativeLayoutPackNr));
    EXPECT_EQ(nat::native_pack_weight(kS4, raw.get(), c.K(), c.N, o.ldb,
                      nat::kNativeLayoutPackNr, c.transB, out.data(), 1),
            status_t::success);
    return out;
}

/// matmul_direct takes the call for the reference algorithm: f16 weight
/// and source scales it takes for the native BRGEMM only.
bool reference_takes(const case_t &c) {
    return !c.direct && c.wei_scale_dt != kF16 && c.src_scale_dt != kF16;
}

/// The default route runs the call on its custom kernel: the fused order (a
/// per-token or per-group source scale, alpha 1, beta 0, an f32 or bf16
/// dst), groups of whole K octets, more than one group, N a multiple of 32,
/// source and weight scales of one type (f32 or bf16), no post-op or gated
/// activation.
bool ck_takes(const case_t &c) {
    return !c.direct && c.fused() && c.gs % 8 == 0 && c.G > 1 && c.N % 32 == 0
            && c.src_scale_dt == c.wei_scale_dt && c.wei_scale_dt != kF16
            && c.po == po_t::none && c.gated == mm::matmul_gated_act_t::none
            && !c.transA && c.ldb_pad == 0;
}

/// The custom kernel gives the bits of the fused order, except with a
/// per-token (or {1, 1}) source scale and a bias: the custom kernel may fuse
/// the source scale's product and the bias into one FMA, on some rows of a
/// call and not on others.
bool ck_keeps_bits(const case_t &c) {
    return c.bias == kNone || c.src_groups();
}

class NativeBrgemmInt4Calls : public ::testing::TestWithParam<case_t> {
protected:
    void SetUp() override {
        if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
        mm::clear_matmul_weight_caches();
    }
    void TearDown() override { mm::clear_matmul_weight_caches(); }
    WeightCacheGuard global_cache_ {2};
};

TEST_P(NativeBrgemmInt4Calls, MatchTheirOrder) {
    const case_t &c = GetParam();
    const operands_t o = make_operands(c);
    if (c.batches) { EXPECT_TRUE(tiles_take_batches(c, o, c.odd_batches)); }
    if (c.passes) { EXPECT_TRUE(tiles_take_passes(c, o)); }
    const buffer_t raw = copy_of(o.wei);
    const source_t s = source_of(c, o);
    // The model reads the raw weight, which an in-place call rewrites.
    const buffer_t model = model_dst(c, o, s);
    const buffer_t got = run_native(c, o);
    if (HasFailure()) return;
    const int cols = c.gated != mm::matmul_gated_act_t::none ? c.N / 2 : c.N;

    // the model of the call's order
    if (transcendental(c))
        EXPECT_TRUE(near_model(c, o.ldc, got.get(), model.get()))
                << "vs the model";
    else
        EXPECT_TRUE(nt::within_steps(
                got.get(), o.ldc, model.get(), o.ldc, c.dst, c.M, cols, 0))
                << "vs the model of the " << (c.fused() ? "fused" : "reference")
                << " order";
    if (in_place(c)) {
        EXPECT_TRUE(nat::native_weight_packed_in_place(o.wei.get(), raw.bytes));
        const std::vector<uint8_t> image = layout_of(c, o, raw);
        ASSERT_EQ(image.size(), raw.bytes);
        EXPECT_EQ(std::memcmp(o.wei.get(), image.data(), raw.bytes), 0)
                << "the buffer does not hold the layout";
    } else {
        EXPECT_EQ(std::memcmp(raw.get(), o.wei.get(), raw.bytes), 0)
                << "the raw weight changed";
    }

    // the reference algorithm, in its order
    if (!c.fused() && reference_takes(c) && !transcendental(c)) {
        buffer_t ref = copy_of(o.init);
        const buffer_t scale = copy_of(o.src_scale);
        nt::call_t rc = make_call(c, o, raw.get(), ref.get(), scale);
        rc.params.lowoha_algo = matmul_algo_t::reference;
        rc.params.weight_cache_type = 0;
        rc.params.num_threads = 0;
        ASSERT_EQ(nt::run_matmul(rc), status_t::success) << "reference";
        EXPECT_TRUE(nt::within_steps(
                got.get(), o.ldc, ref.get(), o.ldc, c.dst, c.M, cols, 0))
                << "vs the reference algorithm";
    }

    // the default route's custom kernel, in the fused order
    if (!ck_takes(c) || !ck_keeps_bits(c)) return;
    if (!nt::default_route_reachable()) {
        RecordProperty("ck", "not_run (process-wide algorithm)");
        return;
    }
    buffer_t ck_dst = copy_of(o.init);
    buffer_t ck_wei = nt::kept_until_exit(copy_of(raw));
    const buffer_t scale = copy_of(o.src_scale);
    nt::call_t kc = make_call(c, o, ck_wei.get(), ck_dst.get(), scale);
    kc.params.weight_cache_type = 2;
    kc.params.num_threads = 0;
    ::reset_grp_matmul_caches();
    {
        nt::ck_route_pin_t route;
        nt::ck_engagement_t engagement;
        ASSERT_EQ(nt::run_matmul(kc), status_t::success) << "default route";
        EXPECT_TRUE(engagement.engaged())
                << "the custom kernel did not serve the call on the default "
                   "route";
        EXPECT_TRUE(nt::within_steps(
                got.get(), o.ldc, ck_dst.get(), o.ldc, c.dst, c.M, cols, 0))
                << "vs the custom kernel of the default route";
        RecordProperty("ck", engagement.engaged() ? "served" : "not_served");
    }
    ::reset_grp_matmul_caches();
}

std::vector<case_t> named(std::vector<case_t> v) {
    std::vector<std::string> seen;
    for (case_t &c : v) {
        std::string n = describe(c);
        int dup = 0;
        for (const std::string &s : seen)
            dup += s == n || s.rfind(n + "_dup", 0) == 0;
        if (dup) n += "_dup" + std::to_string(dup);
        seen.push_back(n);
        c.name = n;
    }
    return v;
}

std::vector<case_t> call_cases() {
    std::vector<case_t> v;
    const auto add = [&v](case_t c) { v.push_back(c); };
    // Group sizes: whole octets, odd quads (4, 12, 20) and one group over K.
    for (int gs : {4, 8, 12, 16, 20, 32, 64, 128, 256, 512})
        for (gran_t gran : {gran_t::token, gran_t::group}) {
            case_t c;
            c.gs = gs;
            c.G = gs >= 256 ? 2 : (gs >= 64 ? 3 : 5);
            c.gran = gran;
            c.M = gs == 32 ? 33 : 7;
            add(c);
        }
    for (int K : {4, 12, 36, 136, 4100}) {
        case_t c;
        c.gs = K;
        c.G = 1;
        c.M = 5;
        c.N = 48;
        add(c);
        c.gran = gran_t::group;
        add(c);
    }
    // Rows, columns and their tails.
    for (int M : {1, 2, 5, 12, 13, 16, 33, 55, 64, 65, 128})
        for (int N : {16, 64, 100}) {
            case_t c;
            c.M = M;
            c.N = N;
            c.gs = 128;
            c.G = 2;
            add(c);
        }
    for (int N : {1, 15, 17, 31, 33}) {
        case_t c;
        c.N = N;
        c.M = 9;
        add(c);
    }
    // One source row of per-group scales, quantized on the calling thread:
    // fewer and more groups than a team of every thread, from a bf16 or an
    // f32 source.
    for (int G : {30, 40})
        for (src_t src : {src_t::bf16, src_t::f32}) {
            case_t c;
            c.M = 1;
            c.N = 64;
            c.gs = 128;
            c.G = G;
            c.gran = gran_t::group;
            c.src = src;
            c.dst = src == src_t::f32 ? kF32 : kBf16;
            c.threads = 0;
            add(c);
        }
    // Sources, their scales and the weight scales.
    for (src_t src : {src_t::bf16, src_t::s8})
        for (gran_t gran : {gran_t::tensor, gran_t::token, gran_t::group})
            for (data_type_t ss : {kF32, kBf16})
                for (data_type_t ws : {kF32, kBf16}) {
                    case_t c;
                    c.src = src;
                    c.gran = gran;
                    c.src_scale_dt = ss;
                    c.wei_scale_dt = ws;
                    c.M = gran == gran_t::tensor ? 1 : 11;
                    add(c);
                }
    // Storage of the operands.
    for (bool transB : {false, true})
        for (int pad : {0, 1, 3})
            for (bool transA : {false, true}) {
                case_t c;
                c.transB = transB;
                c.ldb_pad = pad;
                c.transA = transA;
                c.M = 10;
                c.N = 96;
                add(c);
            }
    // A transposed source the call quantizes, in the reference order, which
    // the reference algorithm gives bit for bit: it quantizes the source as
    // given.
    for (gran_t gran : {gran_t::token, gran_t::group})
        for (data_type_t ss : {kF32, kBf16}) {
            case_t c;
            c.transA = true;
            c.gran = gran;
            c.src_scale_dt = ss;
            c.alpha = 2.0f;
            c.M = 10;
            c.N = 96;
            add(c);
        }
    // Bias, post-ops and the epilogue of either order; into an f32 dst too,
    // which keeps the rounding of the source scale's product before the
    // bias.  The f16-bias cases call the native BRGEMM directly, on any CPU
    // of its instruction sets.
    for (data_type_t b : {kF32, kBf16, kF16})
        for (gran_t gran : {gran_t::token, gran_t::group}) {
            case_t c;
            c.bias = b;
            c.gran = gran;
            c.M = 20;
            c.direct = b == kF16;
            add(c);
            if (c.direct) continue;
            c.dst = kF32;
            add(c);
        }
    for (po_t po : {po_t::add, po_t::gelu})
        for (gran_t gran : {gran_t::token, gran_t::group}) {
            case_t c;
            c.po = po;
            c.gran = gran;
            c.bias = kBf16;
            add(c);
        }
    {
        case_t c;
        c.alpha = 2.0f;
        add(c);
        c.alpha = 0.5f;
        c.beta = 0.25f;
        c.dst = kF32;
        c.direct = true;
        add(c);
        c.alpha = 1.0f;
        c.beta = 1.0f;
        c.gran = gran_t::group;
        add(c);
    }
    // alpha over one group over K before the source and weight scales, beta
    // after them into a float dst and before them into an integer one, and
    // both over groups after their sum, against the reference
    // algorithm: a static or a dynamic source, per tensor at M 1, per token
    // or per group of one, a bf16 or f32 dst, a dst of padded columns, and
    // an integer dst (with the checks off); and alpha over one group of the
    // paired tiles of a gated activation, against the model.
    {
        case_t c;
        c.src = src_t::s8;
        c.M = 1;
        c.N = 32;
        c.gs = 4;
        c.G = 1;
        c.gran = gran_t::tensor;
        c.dst = kF32;
        c.beta = 1.0f;
        add(c);
        c.M = 5;
        c.N = 48;
        c.gs = 136;
        c.gran = gran_t::token;
        c.alpha = 0.5f;
        c.beta = 0.25f;
        add(c);
        c.src = src_t::bf16;
        c.dst = kBf16;
        add(c);
        c.gran = gran_t::group;
        c.N = 40;
        c.alpha = 2.0f;
        c.beta = 0.0f;
        c.bias = kBf16;
        add(c);
        c.gran = gran_t::token;
        c.N = 48;
        c.alpha = 1.0f;
        c.beta = 1.0f;
        c.bias = kNone;
        c.dst = kS8;
        c.direct = true;
        add(c);
        case_t g;
        g.src = src_t::s8;
        g.M = 5;
        g.N = 48;
        g.gs = 32;
        g.G = 4;
        g.dst = kF32;
        g.beta = 1.0f;
        add(g);
        g.src = src_t::bf16;
        g.gran = gran_t::group;
        g.alpha = 0.5f;
        g.beta = 0.25f;
        add(g);
        case_t q;
        q.G = 1;
        q.gs = 64;
        q.M = 7;
        q.N = 128;
        q.alpha = 1.5f;
        q.gated = mm::matmul_gated_act_t::silu_and_mul;
        q.bias = kBf16;
        add(q);
        q.dst = kF32;
        q.gated = mm::matmul_gated_act_t::swiglu_oai_mul;
        add(q);
        q.N = 80;
        q.gated = mm::matmul_gated_act_t::none;
        q.beta = 0.5f;
        q.alpha = 1.0f;
        add(q);
    }
    // Gated activations over the plain layout: [gate | up] halves paired in
    // the tiles (N a multiple of 64) or staged rows (every other even N),
    // and interleaved pairs.
    for (mm::matmul_gated_act_t act : {mm::matmul_gated_act_t::silu_and_mul,
                 mm::matmul_gated_act_t::gelu_and_mul,
                 mm::matmul_gated_act_t::swiglu_oai_mul})
        for (gran_t gran : {gran_t::token, gran_t::group}) {
            case_t c;
            c.gated = act;
            c.gran = gran;
            c.N = 128;
            c.M = 7;
            c.bias = kBf16;
            add(c);
        }
    for (mm::matmul_gated_act_t act : {mm::matmul_gated_act_t::silu_and_mul,
                 mm::matmul_gated_act_t::gelu_and_mul,
                 mm::matmul_gated_act_t::gelu_tanh_and_mul,
                 mm::matmul_gated_act_t::swiglu_oai_mul})
        for (int N = 2; N <= 66; N += 2) {
            case_t c;
            c.gated = act;
            c.N = N;
            c.M = 5;
            add(c);
        }
    // An f32 source or dst, which matmul_direct takes, and f16 weight and
    // source scales, which it takes for the native BRGEMM; integer
    // destinations and an f32 source with a bf16 dst, which it does not take,
    // through native_brgemm_execute.
    for (data_type_t dst : {kF32, kBf16, kS8, kU8, kS32})
        for (src_t src : {src_t::bf16, src_t::f32}) {
            const bool taken
                    = dst == kF32 || (dst == kBf16 && src == src_t::bf16);
            case_t c;
            c.src = src;
            c.dst = dst;
            c.direct = !taken;
            c.M = 9;
            add(c);
            c.wei_scale_dt = kF16;
            c.gran = gran_t::group;
            add(c);
            c.src_scale_dt = kF16;
            add(c);
            c.gran = gran_t::token;
            add(c);
        }
    for (data_type_t ws : {kF32, kBf16, kF16}) {
        case_t c;
        c.src = src_t::s8;
        c.dst = kF32;
        c.wei_scale_dt = ws;
        c.M = 6;
        add(c);
    }
    for (gran_t gran : {gran_t::token, gran_t::group})
        for (data_type_t ws : {kF32, kF16}) {
            case_t c;
            c.src = src_t::s8;
            c.gran = gran;
            c.src_scale_dt = kF16;
            c.wei_scale_dt = ws;
            c.M = 7;
            add(c);
        }
    // Weight cache modes, repeated calls, threads.
    for (int wc : {0, 1, 2})
        for (int threads : {1, 7, 0}) {
            case_t c;
            c.wc = wc;
            c.threads = threads;
            c.calls = 2;
            c.M = 40;
            c.N = 256;
            c.gs = 64;
            c.G = 6;
            add(c);
        }
    // Groups in batches, between which C carries the sum and after the last
    // of which the source scale and the store come.  Many groups of a slab
    // within twice the pass budget, in one batch.  Scales read in place past
    // twice the budget, in batches of whole octets (groups of 2 K quads) and
    // of an odd number of quads (groups of 3, at weight cache 2 over the
    // image).  Scales gathered 32 groups at a time over 100 groups: the last
    // 8 columns of N 40; and the same groups over the paired tiles of a gated
    // call, which read them in place.
    for (gran_t gran : {gran_t::token, gran_t::group}) {
        case_t c;
        c.gran = gran;
        c.gs = 8;
        c.G = 600;
        c.M = 24;
        c.N = 64;
        add(c);
        c.G = 1000;
        c.batches = true;
        add(c);
        c.gs = 12;
        c.G = 500;
        c.wc = 2;
        c.odd_batches = true;
        add(c);
        case_t g;
        g.gran = gran;
        g.gs = 8;
        g.G = 100;
        g.M = 7;
        g.N = 40;
        add(g);
        g.N = 128;
        g.gated = mm::matmul_gated_act_t::silu_and_mul;
        add(g);
    }
    // One group over K in K passes, between which C carries the exact int32
    // sum, on one thread (a group whose slab exceeds the int8_l2_pct share
    // of L2): whole octets, and an odd number of quads over a dst of padded
    // columns (K 65540, N 48); the fused order into a bf16 or f32 dst, from
    // a dynamic bf16 or f32 or a static s8 source, at weight cache 2 over
    // the image; alpha and beta before the scales, and an integer dst, in
    // the reference order.
    {
        case_t c;
        c.gs = 65536;
        c.G = 1;
        c.M = 24;
        c.N = 64;
        c.threads = 1;
        c.passes = true;
        add(c);
        c.dst = kF32;
        c.src = src_t::f32;
        add(c);
        c.src = src_t::s8;
        c.wc = 2;
        add(c);
        c.src = src_t::bf16;
        c.wc = 1;
        c.gs = 65540;
        c.M = 25;
        c.N = 48;
        c.dst = kBf16;
        add(c);
        c.gs = 65536;
        c.M = 20;
        c.N = 64;
        c.dst = kF32;
        c.alpha = 0.5f;
        c.beta = 0.25f;
        c.direct = true;
        add(c);
        c.alpha = 1.0f;
        c.beta = 1.0f;
        c.dst = kS8;
        add(c);
    }
    // Prompt-sized calls.
    for (int M : {256, 1024}) {
        case_t c;
        c.M = M;
        c.N = 128;
        c.gs = 128;
        c.G = 4;
        c.threads = 0;
        add(c);
    }
    // Every call the custom kernel takes into an f32 dst as well, which keeps
    // the order of the group sums that a bf16 store rounds away.
    const size_t taken = v.size();
    for (size_t i = 0; i < taken; ++i) {
        if (!ck_takes(v[i]) || v[i].dst != kBf16) continue;
        case_t c = v[i];
        c.dst = kF32;
        const std::string d = describe(c);
        if (std::none_of(v.begin(), v.end(),
                    [&d](const case_t &x) { return describe(x) == d; }))
            add(c);
    }
    return named(v);
}

INSTANTIATE_TEST_SUITE_P(NativeBrgemmInt4Calls, NativeBrgemmInt4Calls,
        ::testing::ValuesIn(call_cases()),
        [](const ::testing::TestParamInfo<case_t> &info) {
            return info.param.name;
        });

// The plan line names the order: fused for a per-token or per-group source,
// reference for alpha, beta or an integer dst.
TEST(NativeBrgemmInt4Decision, OrderOfTheCall) {
    case_t c;
    c.name = "decision";
    const operands_t o = make_operands(c);
    const buffer_t scale = copy_of(o.src_scale);
    nt::call_t call = make_call(c, o, o.wei.get(), nullptr, scale);
    const auto decide = [](const nt::call_t &k) {
        nat::GemmDescriptor d;
        d.M = k.M;
        d.N = k.N;
        d.K = k.K;
        d.alpha = k.alpha;
        d.beta = k.beta;
        d.src_dt = k.params.dtypes.src;
        d.wei_dt = k.params.dtypes.wei;
        d.dst_dt = k.params.dtypes.dst;
        return nat::decide_int8_call(d, k.params, k.bias);
    };
    nat::native_int8_decision_t d = decide(call);
    EXPECT_EQ(d.scheme, nat::native_int8_scheme_t::per_group);
    EXPECT_EQ(d.groups, c.G);
    EXPECT_EQ(d.order, nat::native_int8_order_t::fused);
    EXPECT_TRUE(d.direct_store);
    EXPECT_FALSE(d.bias_in_dequant);
    call.params.quant_params.src_scale.dims = {c.M, c.G};
    d = decide(call);
    EXPECT_EQ(d.order, nat::native_int8_order_t::fused);
    EXPECT_TRUE(d.direct_store);
    call.params.quant_params.src_scale.dims = {c.M, 1};
    call.alpha = 2.0f;
    d = decide(call);
    EXPECT_EQ(d.order, nat::native_int8_order_t::reference);
    EXPECT_FALSE(d.direct_store);
    call.alpha = 1.0f;
    call.params.dtypes.dst = kS8;
    d = decide(call);
    EXPECT_EQ(d.order, nat::native_int8_order_t::reference);
    EXPECT_FALSE(d.scales_after_groups);
    call.beta = 1.0f;
    d = decide(call);
    EXPECT_EQ(d.order, nat::native_int8_order_t::reference);
    EXPECT_FALSE(d.scales_after_groups) << "alpha and beta after the groups";
    call.beta = 0.0f;
    // One group over K: alpha before both scales, and beta before or after
    // them by the dst.
    call.params.dtypes.dst = kBf16;
    call.params.quant_params.wei_scale.dims = {1, c.N};
    d = decide(call);
    EXPECT_EQ(d.groups, 1);
    EXPECT_EQ(d.order, nat::native_int8_order_t::fused);
    EXPECT_FALSE(d.scales_after_groups);
    call.params.dtypes.dst = kS8;
    d = decide(call);
    EXPECT_EQ(d.order, nat::native_int8_order_t::reference);
    EXPECT_FALSE(d.scales_after_groups);
    for (const float alpha : {1.0f, 2.0f})
        for (const float beta : {0.0f, 1.0f}) {
            if (alpha == 1.0f && beta == 0.0f) continue;
            call.alpha = alpha;
            call.beta = beta;
            d = decide(call);
            EXPECT_EQ(d.order, nat::native_int8_order_t::reference);
            EXPECT_TRUE(d.scales_after_groups) << alpha << " " << beta;
            EXPECT_FALSE(d.direct_store);
        }
    call.alpha = 1.0f;
    call.beta = 0.0f;
    call.params.dtypes.dst = kBf16;
    call.params.quant_params.wei_scale.dims = {c.N};
    EXPECT_EQ(decide(call).groups, 0);
}

// The problem of an s4 call names its group size, K for one group over K,
// and the bytes of its weight scales: calls of other group sizes or scale
// types are other problems.  Other weights name none.
TEST(NativeBrgemmInt4Decision, ProblemOfTheCall) {
    case_t c;
    c.name = "problem";
    const operands_t o = make_operands(c);
    const buffer_t scale = copy_of(o.src_scale);
    const nt::call_t call = make_call(c, o, o.wei.get(), nullptr, scale);
    const auto problem = [](const nt::call_t &k) {
        nat::GemmDescriptor d;
        d.M = k.M;
        d.N = k.N;
        d.K = k.K;
        d.alpha = k.alpha;
        d.beta = k.beta;
        d.src_dt = k.params.dtypes.src;
        d.wei_dt = k.params.dtypes.wei;
        d.dst_dt = k.params.dtypes.dst;
        return nat::make_native_problem(d, k.params, k.bias,
                nat::kNativeLayoutPackNr,
                nat::decide_int8_call(d, k.params, k.bias));
    };
    const nat::native_problem_t p = problem(call);
    EXPECT_TRUE(p.int4());
    EXPECT_TRUE(p.int8());
    EXPECT_EQ(p.int8_scheme, nat::native_int8_scheme_t::per_group);
    EXPECT_EQ(p.wei_group_size, c.gs);
    EXPECT_EQ(p.wei_scale_bytes, 2);
    nt::call_t f32_scales = call;
    f32_scales.params.quant_params.wei_scale.dt = kF32;
    const nat::native_problem_t f = problem(f32_scales);
    EXPECT_EQ(f.wei_scale_bytes, 4);
    EXPECT_FALSE(p == f);
    nt::call_t one = call;
    one.params.quant_params.wei_scale.dims = {1, c.N};
    const nat::native_problem_t q = problem(one);
    EXPECT_EQ(q.wei_group_size, c.K());
    EXPECT_FALSE(p == q);
    EXPECT_NE(nat::hash_native_problem(p), nat::hash_native_problem(q));
    nat::native_problem_t same = q;
    same.wei_group_size = c.gs;
    EXPECT_TRUE(p == same);
    EXPECT_EQ(nat::hash_native_problem(p), nat::hash_native_problem(same));
    nt::call_t s8 = call;
    s8.params.dtypes.wei = kS8;
    const nat::native_problem_t r = problem(s8);
    EXPECT_FALSE(r.int4());
    EXPECT_EQ(r.int8_scheme, nat::native_int8_scheme_t::per_group);
    EXPECT_EQ(r.wei_group_size, 0);
    EXPECT_EQ(r.wei_scale_bytes, 0);
}

// The layout rule takes an s4 weight with an s8 source (or one the call
// quantizes to s8) over groups of whole K quads, and names why it declines
// the others.
TEST(NativeBrgemmInt4LayoutRule, TakesAndDeclines) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    nat::native_layout_query_t q;
    q.wei_dt = kS4;
    q.src_dt = kS8;
    q.K = 4096;
    q.N = 100;
    q.ldb = 4096;
    q.transB = true;
    q.wei_group_size = 128;
    EXPECT_TRUE(nat::decide_native_layout(q).native);
    q.wei_group_size = 0;
    EXPECT_TRUE(nat::decide_native_layout(q).native) << "one group over K";
    q.wei_group_size = 12;
    q.K = 36;
    q.ldb = 36;
    EXPECT_TRUE(nat::decide_native_layout(q).native) << "odd quads";
    const auto declined = [&](nat::native_layout_query_t x) {
        return std::string(nat::decide_native_layout(x).reason);
    };
    nat::native_layout_query_t u8 = q;
    u8.src_dt = kU8;
    EXPECT_EQ(declined(u8), "a u8 source with an s4 weight");
    nat::native_layout_query_t bf16 = q;
    bf16.src_dt = kBf16;
    EXPECT_EQ(declined(bf16), "s4 weight with a source other than s8");
    nat::native_layout_query_t six = q;
    six.wei_group_size = 6;
    EXPECT_EQ(declined(six),
            "an s4 weight in groups of other than whole K quads");
    nat::native_layout_query_t uneven = q;
    uneven.wei_group_size = -1;
    EXPECT_EQ(declined(uneven),
            "an s4 weight in groups of other than whole K quads");
    nat::native_layout_query_t big = q;
    big.K = (nat::kMaxSafeInt4GroupK / 4 + 1) * 4;
    big.ldb = big.K;
    big.wei_group_size = 0;
    EXPECT_EQ(declined(big),
            "an s4 weight group above the int32 accumulation bound");
    nat::native_layout_query_t u4 = q;
    u4.wei_dt = data_type_t::u4;
    EXPECT_EQ(
            declined(u4), "weight type other than bf16 / s8 / s4 / f32 / f16");
}

// matmul_direct's checks take f16 weight scales of a DA8W4 call only when
// the call asks for the native BRGEMM (the process-wide ALGO 10, or
// lowoha_algo native_brgemm on one call): every other call is refused, a
// batched call or one that names another algorithm under ALGO 10 too.  An
// f32 source or dst is taken on every route; an f32 source with a bf16 dst
// on none, as for any call.
TEST(NativeBrgemmInt4Validation, F16ScalesOnlyOnTheNativeBrgemm) {
    struct combo_t {
        src_t src;
        data_type_t dst, wei_scale_dt;
        bool native, other; ///< taken by the native BRGEMM / elsewhere
    };
    const combo_t combos[] = {{src_t::bf16, kBf16, kBf16, true, true},
            {src_t::bf16, kF32, kBf16, true, true},
            {src_t::f32, kF32, kBf16, true, true},
            {src_t::f32, kF32, kF32, true, true},
            {src_t::f32, kBf16, kBf16, false, false},
            {src_t::s8, kBf16, kBf16, true, true},
            {src_t::s8, kF32, kF32, true, true},
            {src_t::bf16, kBf16, kF16, true, false},
            {src_t::f32, kF32, kF16, true, false},
            {src_t::s8, kF32, kF16, true, false}};
    enum class route_t { unset, per_call, process, batched, other_algo };
    for (const combo_t &k : combos)
        for (const route_t route : {route_t::unset, route_t::per_call,
                     route_t::process, route_t::batched, route_t::other_algo}) {
            case_t c;
            c.src = k.src;
            c.dst = k.dst;
            c.wei_scale_dt = k.wei_scale_dt;
            c.name = describe(c) + "_route"
                    + std::to_string(static_cast<int>(route));
            SCOPED_TRACE(c.name);
            const operands_t o = make_operands(c);
            buffer_t dst = copy_of(o.init);
            mm::matmul_params p = o.params;
            const bool process = route == route_t::process
                    || route == route_t::batched
                    || route == route_t::other_algo;
            const nt::algo_pin_t pin(process ? matmul_algo_t::native_brgemm
                                             : matmul_algo_t::none);
            if (route == route_t::per_call)
                p.lowoha_algo = matmul_algo_t::native_brgemm;
            if (route == route_t::other_algo)
                p.lowoha_algo = matmul_algo_t::reference;
            const int batch = route == route_t::batched ? 2 : 1;
            const bool native
                    = route == route_t::per_call || route == route_t::process;
            EXPECT_EQ(
                    mm::validate_matmul_direct_inputs(o.src.get(), o.wei.get(),
                            dst.get(), c.M, c.N, c.K(), batch, batch, p, true),
                    (native ? k.native : k.other) ? status_t::success
                                                  : status_t::failure);
        }
}

// At weight cache 2 the first call on a constant s4 weight whose layout is
// its own size rewrites it in place under the process-wide ALGO 10; a call
// that asks for the native BRGEMM by itself caches the layout out of place
// and leaves the weight raw, and so does any call on a layout larger than
// the buffer (K an odd number of quads, N not a multiple of 32, rows past
// their length).  Every call, and a second one on the image, gives the bits
// of a call on the cached layout.
TEST(NativeBrgemmInt4InPlace, OnlyUnderTheProcessWideAlgorithm) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    mm::clear_matmul_weight_caches();
    const WeightCacheGuard cache(2);
    struct shape_t {
        int N, gs, G, ldb_pad;
        bool fits;
    };
    for (const shape_t &sh : {shape_t {64, 32, 4, 0, true},
                 shape_t {64, 12, 3, 0, false}, shape_t {48, 32, 4, 0, false},
                 shape_t {64, 32, 4, 2, false}})
        for (const bool per_call : {false, true}) {
            case_t c;
            c.M = 8;
            c.N = sh.N;
            c.gs = sh.gs;
            c.G = sh.G;
            c.ldb_pad = sh.ldb_pad;
            c.wc = 2;
            c.name = describe(c) + (per_call ? "_per_call" : "_process");
            SCOPED_TRACE(c.name);
            const operands_t o = make_operands(c);
            const buffer_t raw = copy_of(o.wei);
            buffer_t want = copy_of(o.init);
            {
                const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
                const buffer_t w = nt::kept_until_exit(copy_of(raw));
                const buffer_t scale = copy_of(o.src_scale);
                nt::call_t k = make_call(c, o, w.get(), want.get(), scale);
                k.params.weight_cache_type = 1;
                ASSERT_EQ(nt::run_matmul(k), status_t::success);
            }
            const nt::algo_pin_t pin(per_call ? matmul_algo_t::none
                                              : matmul_algo_t::native_brgemm);
            for (int call = 0; call < 2; ++call) {
                buffer_t got = copy_of(o.init);
                const buffer_t scale = copy_of(o.src_scale);
                nt::call_t k = make_call(c, o, o.wei.get(), got.get(), scale);
                if (per_call)
                    k.params.lowoha_algo = matmul_algo_t::native_brgemm;
                ASSERT_EQ(nt::run_matmul(k), status_t::success) << call;
                EXPECT_TRUE(nt::within_steps(got.get(), o.ldc, want.get(),
                        o.ldc, c.dst, c.M, c.N, 0))
                        << "call " << call;
            }
            const bool image = sh.fits && !per_call;
            EXPECT_EQ(
                    nat::native_weight_packed_in_place(o.wei.get(), raw.bytes),
                    image);
            if (image) {
                const std::vector<uint8_t> layout = layout_of(c, o, raw);
                ASSERT_EQ(layout.size(), raw.bytes);
                EXPECT_EQ(
                        std::memcmp(o.wei.get(), layout.data(), raw.bytes), 0);
            } else {
                EXPECT_EQ(std::memcmp(o.wei.get(), raw.get(), raw.bytes), 0)
                        << "the raw weight changed";
            }
        }
    mm::clear_matmul_weight_caches();
}

// A weight of K an odd number of quads, whose layout is larger than the raw
// weight, is rewritten in place at weight cache 2 when the call states a
// buffer of at least the layout's size (wei_buffer_capacity_bytes), with
// nothing written past the layout; a stated buffer of one byte less keeps
// the weight raw.  Every call, and a second one, gives the bits of a call on
// the cached layout.
TEST(NativeBrgemmInt4InPlace, StatedCapacityTakesALargerLayout) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    mm::clear_matmul_weight_caches();
    const WeightCacheGuard cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    case_t c;
    c.M = 8;
    c.N = 64;
    c.gs = 12;
    c.G = 3;
    c.wc = 2;
    c.name = describe(c) + "_capacity";
    const operands_t o = make_operands(c);
    const buffer_t raw = copy_of(o.wei);
    const std::vector<uint8_t> layout = layout_of(c, o, raw);
    ASSERT_GT(layout.size(), raw.bytes);
    buffer_t want = copy_of(o.init);
    {
        const buffer_t w = nt::kept_until_exit(copy_of(raw));
        const buffer_t scale = copy_of(o.src_scale);
        nt::call_t k = make_call(c, o, w.get(), want.get(), scale);
        k.params.weight_cache_type = 1;
        ASSERT_EQ(nt::run_matmul(k), status_t::success);
    }
    constexpr uint8_t kCanary = 0xa5;
    for (const size_t capacity : {layout.size(), layout.size() - 1}) {
        SCOPED_TRACE("capacity " + std::to_string(capacity));
        const bool fits = capacity >= layout.size();
        const buffer_t w
                = nt::kept_until_exit(nt::make_buffer(layout.size() + 64));
        std::memset(w.get(), kCanary, w.bytes);
        std::memcpy(w.get(), raw.get(), raw.bytes);
        for (int call = 0; call < 2; ++call) {
            buffer_t got = copy_of(o.init);
            const buffer_t scale = copy_of(o.src_scale);
            nt::call_t k = make_call(c, o, w.get(), got.get(), scale);
            k.params.wei_buffer_capacity_bytes = capacity;
            ASSERT_EQ(nt::run_matmul(k), status_t::success) << call;
            EXPECT_TRUE(nt::within_steps(
                    got.get(), o.ldc, want.get(), o.ldc, c.dst, c.M, c.N, 0))
                    << "call " << call;
        }
        EXPECT_EQ(nat::native_weight_packed_in_place(w.get(), capacity), fits);
        const uint8_t *b = w.as<uint8_t>();
        if (fits)
            EXPECT_EQ(std::memcmp(b, layout.data(), layout.size()), 0)
                    << "the buffer does not hold the layout";
        else
            EXPECT_EQ(std::memcmp(b, raw.get(), raw.bytes), 0)
                    << "the raw weight changed";
        const size_t kept = fits ? layout.size() : raw.bytes;
        EXPECT_EQ(std::count(b + kept, b + w.bytes, kCanary),
                static_cast<std::ptrdiff_t>(w.bytes - kept))
                << "bytes past the " << (fits ? "layout" : "raw weight")
                << " written";
    }
    mm::clear_matmul_weight_caches();
}

// An s4 image has no other reader: a call on it that the native BRGEMM does
// not serve, or that reads it as another weight, fails with dst untouched
// and the image kept.
TEST(NativeBrgemmInt4InPlace, TheImageHasNoOtherReader) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    mm::clear_matmul_weight_caches();
    const WeightCacheGuard cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    case_t c;
    c.name = "image_readers";
    c.M = 4;
    c.N = 64;
    c.gs = 32;
    c.G = 8;
    c.wc = 2;
    const operands_t o = make_operands(c);
    {
        buffer_t dst = copy_of(o.init);
        const buffer_t scale = copy_of(o.src_scale);
        ASSERT_EQ(
                nt::run_matmul(make_call(c, o, o.wei.get(), dst.get(), scale)),
                status_t::success);
    }
    ASSERT_TRUE(nat::native_weight_packed_in_place(o.wei.get(), o.wei.bytes));
    const buffer_t image = copy_of(o.wei);
    // The same bytes as a weight of 32 columns over twice the K.
    case_t other = c;
    other.name = "image_readers_other";
    other.N = 32;
    other.G = 16;
    const operands_t q = make_operands(other);
    ASSERT_EQ(q.wei.bytes, o.wei.bytes);
    struct variant_t {
        const char *name;
        nt::call_t call;
    };
    std::vector<variant_t> variants;
    buffer_t dst = copy_of(o.init);
    buffer_t other_dst = copy_of(q.init);
    const buffer_t scale = copy_of(o.src_scale);
    const buffer_t other_scale = copy_of(q.src_scale);
    const nt::call_t good = make_call(c, o, o.wei.get(), dst.get(), scale);
    variants.push_back({"another weight",
            make_call(other, q, o.wei.get(), other_dst.get(), other_scale)});
    nt::call_t u8 = good;
    u8.params.dtypes.compute = kU8;
    variants.push_back({"a u8 source", u8});
    nt::call_t kn = good;
    kn.transB = false;
    kn.ldb = c.N;
    variants.push_back({"the weight read as [K, N]", kn});
    nt::call_t reordered = good;
    reordered.params.mem_format_a = 'r';
    variants.push_back({"a reordered source", reordered});
    for (const variant_t &v : variants) {
        SCOPED_TRACE(v.name);
        EXPECT_EQ(nt::run_matmul(v.call), status_t::failure);
        EXPECT_TRUE(
                nat::native_weight_packed_in_place(o.wei.get(), o.wei.bytes));
        EXPECT_EQ(std::memcmp(o.wei.get(), image.get(), image.bytes), 0)
                << "the image changed";
    }
    EXPECT_EQ(std::memcmp(dst.get(), o.init.get(), dst.bytes), 0)
            << "dst written by a refused call";
    EXPECT_EQ(std::memcmp(other_dst.get(), q.init.get(), other_dst.bytes), 0)
            << "dst written by a refused call";
    mm::clear_matmul_weight_caches();
}

// The calls with an s4 weight that the native BRGEMM takes and does not
// serve: each returns unimplemented from matmul_direct with lowoha_algo
// native_brgemm, with dst untouched and its reason logged (refused_with);
// under ALGO 10 without lowoha_algo the native BRGEMM hands it back.
TEST(NativeBrgemmInt4Refusal, UnservedCallsReturnUnimplemented) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    case_t c;
    c.name = "refused";
    c.M = 4;
    const operands_t o = make_operands(c);
    buffer_t dst = copy_of(o.init);
    const buffer_t scale = copy_of(o.src_scale);
    const nt::call_t good = make_call(c, o, o.wei.get(), dst.get(), scale);
    std::vector<int8_t> zps(c.N, 1);
    {
        // matmul_direct's checks refuse weight zero points first: the
        // native check alone.
        nt::call_t k = good;
        k.params.quant_params.wei_zp.buff = zps.data();
        k.params.quant_params.wei_zp.dt = kS8;
        k.params.quant_params.wei_zp.dims = {1, c.N};
        nat::GemmDescriptor d;
        d.M = k.M;
        d.N = k.N;
        d.K = k.K;
        d.src_dt = k.params.dtypes.src;
        d.wei_dt = kS4;
        d.dst_dt = kBf16;
        const char *reason = nullptr;
        EXPECT_EQ(nat::check_int8_call(d, k.params,
                          nat::decide_int8_call(d, k.params, nullptr), &reason),
                status_t::unimplemented);
        EXPECT_STREQ(reason, "an s4 weight with zero points");
    }
    {
        // A u8 source, quantized in the call.
        nat::GemmDescriptor d;
        d.M = good.M;
        d.N = good.N;
        d.K = good.K;
        d.src_dt = kBf16;
        d.wei_dt = kS4;
        d.dst_dt = kBf16;
        mm::matmul_params p = good.params;
        p.dtypes.compute = kU8;
        const char *reason = nullptr;
        EXPECT_EQ(nat::check_int8_call(
                          d, p, nat::decide_int8_call(d, p, nullptr), &reason),
                status_t::unimplemented);
        EXPECT_STREQ(reason, "a u8 source with an s4 weight");
        p.dtypes.compute = kS8;
        p.quant_params.wei_scale.dims = {c.N};
        EXPECT_EQ(nat::check_int8_call(
                          d, p, nat::decide_int8_call(d, p, nullptr), &reason),
                status_t::unimplemented);
        EXPECT_STREQ(
                reason, "an s4 weight with weight scales other than {G, N}");
        p.quant_params.wei_scale.dims = {c.G, c.N};
        p.quant_params.src_scale.dims = {c.M, 2 * c.G};
        EXPECT_EQ(nat::check_int8_call(
                          d, p, nat::decide_int8_call(d, p, nullptr), &reason),
                status_t::unimplemented)
                << "an {M, G} source scale of other groups";
    }
    // A source zero point, which matmul_direct's checks refuse first: the
    // native check, and native_brgemm_execute as with the checks off, over
    // an s8 source and over one the call quantizes to s8.
    for (const src_t src : {src_t::s8, src_t::bf16}) {
        case_t z = c;
        z.name = src == src_t::s8 ? "refused_zp_s8" : "refused_zp_dynbf16";
        z.src = src;
        SCOPED_TRACE(z.name);
        const operands_t q = make_operands(z);
        buffer_t zdst = copy_of(q.init);
        const buffer_t zscale = copy_of(q.src_scale);
        nt::call_t k = make_call(z, q, q.wei.get(), zdst.get(), zscale);
        const int32_t zp = 3;
        set_quant(k.params.quant_params.src_zp, &zp, kS32, {1, 1});
        nat::GemmDescriptor d;
        d.M = k.M;
        d.N = k.N;
        d.K = k.K;
        d.src_dt = k.params.dtypes.src;
        d.wei_dt = kS4;
        d.dst_dt = k.params.dtypes.dst;
        const char *reason = nullptr;
        EXPECT_EQ(nat::check_int8_call(d, k.params,
                          nat::decide_int8_call(d, k.params, nullptr), &reason),
                status_t::unimplemented);
        EXPECT_STREQ(reason, "an s4 weight with a source zero point");
        EXPECT_EQ(run_direct(k), status_t::unimplemented);
        EXPECT_EQ(std::memcmp(zdst.get(), q.init.get(), zdst.bytes), 0)
                << "dst written by a refused call";
    }
    // Without AOCL-DLP a GGML pack and a reordered weight keep the error:
    // the reference kernel does not read them.
    const status_t handed_back = ZENDNNL_DEPENDS_AOCLDLP
            ? status_t::success
            : status_t::unimplemented;
    {
        nt::call_t k = good;
        k.params.packing.pack_format_b = 1;
        EXPECT_EQ(nt::refusal_of(k), handed_back);
        k.params.lowoha_algo = matmul_algo_t::native_brgemm;
        EXPECT_TRUE(nt::refused_with(k, status_t::unimplemented));
    }
    {
        nt::call_t k = good;
        k.params.mem_format_b = 'r';
        EXPECT_EQ(nt::refusal_of(k), handed_back);
        k.params.lowoha_algo = matmul_algo_t::native_brgemm;
        EXPECT_TRUE(nt::refused_with(k, status_t::unimplemented));
    }
    {
        nt::call_t k = good;
        k.params.lowoha_algo = matmul_algo_t::native_brgemm;
        mm::matmul_batch_params_t batch;
        batch.Batch_A = 2;
        batch.Batch_B = 2;
        const char *why = nullptr;
        EXPECT_EQ(
                mm::native_brgemm_refusal('r', k.transA, k.transB, k.M, k.N,
                        k.K, k.alpha, k.src, k.lda_or_tight(), k.wei,
                        k.ldb_or_tight(), k.bias, k.beta, k.dst,
                        k.ldc_or_tight(), true, batch, k.params, nullptr, &why),
                status_t::unimplemented);
        EXPECT_STREQ(why, "a batched call");
    }
    EXPECT_EQ(std::memcmp(dst.get(), o.init.get(), dst.bytes), 0)
            << "dst written by a refused call";
    EXPECT_EQ(nt::native_layouts(o.wei.get()), 0u);
    mm::clear_matmul_weight_caches();
}

// K up to 2147483640, the largest multiple of 8 an int holds: the layout's
// K octets and the call's source rows padded to whole octets stay within
// an int.  A K past it, in groups the other checks take, is refused by the
// layout rule and by the INT8 check, which run before any work.
TEST(NativeBrgemmInt4Refusal, KWhoseOctetsExceedAnInt) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    const int N = 32, gs = 4;
    const float wei_scale = 1.0f, src_scale = 1.0f;
    for (const int K : {2147483640, 2147483644}) {
        SCOPED_TRACE("K " + std::to_string(K));
        const bool fits = K == 2147483640;
        nat::native_layout_query_t q;
        q.wei_dt = kS4;
        q.src_dt = kS8;
        q.K = K;
        q.N = N;
        q.ldb = K;
        q.transB = true;
        q.wei_group_size = gs;
        const nat::native_layout_decision_t rule = nat::decide_native_layout(q);
        EXPECT_EQ(rule.native, fits);
        if (!fits) {
            EXPECT_STREQ(rule.reason,
                    "an s4 weight of K whose whole octets exceed int");
        }
        nat::GemmDescriptor d;
        d.M = 1;
        d.N = N;
        d.K = K;
        d.lda = K;
        d.ldb = K;
        d.ldc = N;
        d.transB = true;
        d.src_dt = kS8;
        d.wei_dt = kS4;
        d.dst_dt = kF32;
        mm::matmul_params p;
        p.dtypes.src = kS8;
        p.dtypes.wei = kS4;
        p.dtypes.dst = kF32;
        p.dtypes.compute = kS8;
        set_quant(p.quant_params.wei_scale, &wei_scale, kF32, {K / gs, N});
        set_quant(p.quant_params.src_scale, &src_scale, kF32, {1, 1});
        const char *reason = nullptr;
        EXPECT_EQ(nat::check_int8_call(
                          d, p, nat::decide_int8_call(d, p, nullptr), &reason),
                fits ? status_t::success : status_t::unimplemented);
        if (!fits) {
            EXPECT_STREQ(
                    reason, "an s4 weight of K whose whole octets exceed int");
        }
    }
}

// The layout's K groups and block bytes at the largest K an s4 weight takes
// (also rounded up from a K of whole quads only), and of the other weight
// types at the largest int: no intermediate value overflows.
TEST(NativeBrgemmInt4Bounds, KGroupsOfTheLargestK) {
    const int K = nat::kMaxNativeInt4K;
    ASSERT_EQ(K, 2147483640);
    EXPECT_EQ(nat::native_k_groups(kS4, K), 268435455);
    EXPECT_EQ(nat::native_k_groups(kS4, K - 4), 268435455);
    EXPECT_EQ(nat::native_k_groups(kS4, 800000000), 100000000);
    EXPECT_EQ(nat::native_block_bytes(kS4, K, nat::kNativeLayoutPackNr),
            size_t {268435455} * 4 * nat::kNativeLayoutPackNr);
    EXPECT_EQ(nat::native_packed_bytes(kS4, K, 64, nat::kNativeLayoutPackNr),
            size_t {268435455} * 4 * 64);
    const int max = std::numeric_limits<int>::max();
    EXPECT_EQ(nat::native_k_groups(kBf16, max), 1073741824);
    EXPECT_EQ(nat::native_k_groups(kS8, max), 536870912);
    EXPECT_EQ(nat::native_k_groups(kF32, max), max);
}

// A call at the largest K an s4 weight takes, in the largest groups that
// divide it (1247 groups of 1722120 values): the checks take it, and its
// source rows, whole octets already, are kept in place without a copy.
TEST(NativeBrgemmInt4Bounds, SourceRowsOfTheLargestK) {
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    const int K = nat::kMaxNativeInt4K, N = 32, groups = 1247;
    ASSERT_EQ(K % groups, 0);
    ASSERT_LE(K / groups, nat::kMaxSafeInt4GroupK);
    const std::vector<float> wei_scale(static_cast<size_t>(groups) * N, 1.0f);
    const float src_scale = 1.0f;
    // The call keeps the source pointer and reads no row.
    const int8_t src[64] = {};
    nat::GemmDescriptor d;
    d.M = 1;
    d.N = N;
    d.K = K;
    d.lda = K;
    d.ldb = K;
    d.ldc = N;
    d.transB = true;
    d.src_dt = kS8;
    d.wei_dt = kS4;
    d.dst_dt = kF32;
    mm::matmul_params p;
    p.dtypes.src = kS8;
    p.dtypes.wei = kS4;
    p.dtypes.dst = kF32;
    p.dtypes.compute = kS8;
    set_quant(p.quant_params.wei_scale, wei_scale.data(), kF32, {groups, N});
    set_quant(p.quant_params.src_scale, &src_scale, kF32, {1, 1});
    const nat::native_int8_decision_t decision
            = nat::decide_int8_call(d, p, nullptr);
    ASSERT_EQ(nat::check_int8_call(d, p, decision), status_t::success);
    nat::native_int8_call_t call;
    ASSERT_EQ(nat::prepare_int8_call(
                      d, p, decision, src, 64, int64_t {1} << 40, call),
            status_t::success);
    EXPECT_EQ(call.A, reinterpret_cast<const uint8_t *>(src));
    EXPECT_EQ(call.lda, K);
    EXPECT_TRUE(call.a_buf.empty());
    EXPECT_EQ(call.groups, groups);
    EXPECT_EQ(call.group_quads, K / groups / 4);
}

// The same call with a transposed source: the checks take it, and its rows
// are copied (copy_int8_source, as prepare_int8_call copies them) in blocks
// of 64 columns, the last of which starts within 64 of the int bound.  The
// source is a column-major [K, 1] in a sparse mapping, untouched but for its
// last pages; the copy's 2 GiB row is address space whose every 2 MiB maps
// one region of a memory file, so the bytes last written at each offset of
// the region are those of the row's last 2 MiB.
TEST(NativeBrgemmInt4Bounds, TransposedSourceRowsOfTheLargestK) {
#if defined(__linux__)
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    const int K = nat::kMaxNativeInt4K, N = 32, groups = 1247;
    const std::vector<float> wei_scale(static_cast<size_t>(groups) * N, 1.0f);
    const float src_scale = 1.0f;
    nat::GemmDescriptor d;
    d.M = 1;
    d.N = N;
    d.K = K;
    d.lda = 1;
    d.ldb = K;
    d.ldc = N;
    d.transA = true;
    d.transB = true;
    d.src_dt = kS8;
    d.wei_dt = kS4;
    d.dst_dt = kF32;
    mm::matmul_params p;
    p.dtypes.src = kS8;
    p.dtypes.wei = kS4;
    p.dtypes.dst = kF32;
    p.dtypes.compute = kS8;
    set_quant(p.quant_params.wei_scale, wei_scale.data(), kF32, {groups, N});
    set_quant(p.quant_params.src_scale, &src_scale, kF32, {1, 1});
    ASSERT_EQ(nat::check_int8_call(d, p, nat::decide_int8_call(d, p, nullptr)),
            status_t::success);

    constexpr size_t kRegion = size_t {2} << 20;
    constexpr int kTail = 8192;
    const size_t bytes = static_cast<size_t>(K);
    const size_t row_bytes = (bytes + kRegion - 1) / kRegion * kRegion;
    struct mappings_t {
        void *src = MAP_FAILED, *row = MAP_FAILED;
        size_t src_bytes = 0, row_bytes = 0;
        int fd = -1;
        ~mappings_t() {
            if (src != MAP_FAILED) munmap(src, src_bytes);
            if (row != MAP_FAILED) munmap(row, row_bytes);
            if (fd >= 0) close(fd);
        }
    } map;
    map.src_bytes = bytes;
    map.src = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    map.row_bytes = row_bytes;
    map.row = mmap(nullptr, row_bytes, PROT_NONE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    map.fd = memfd_create("int4_bounds_row", 0);
    if (map.src == MAP_FAILED || map.row == MAP_FAILED || map.fd < 0
            || ftruncate(map.fd, kRegion) != 0)
        GTEST_SKIP() << "no address space or memory file for the rows";
    uint8_t *const row = static_cast<uint8_t *>(map.row);
    for (size_t at = 0; at < row_bytes; at += kRegion) {
        if (mmap(row + at, kRegion, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED, map.fd, 0)
                == MAP_FAILED)
            GTEST_SKIP() << "no mapping of the memory file";
    }
    // Column k of the source's last kTail columns holds a nonzero byte of
    // k; the others are 0.
    uint8_t *const src = static_cast<uint8_t *>(map.src);
    const auto value = [](int k) { return static_cast<uint8_t>(1 + k % 251); };
    for (int k = K - kTail; k < K; ++k)
        src[k] = value(k);
    nat::copy_int8_source(src, 1, true, 1, K, row, K, 1, 64, int64_t {1} << 40);
    int wrong = 0;
    for (int k = K - kTail; k < K; ++k)
        wrong += row[k] != value(k);
    EXPECT_EQ(wrong, 0) << "of the source's last " << kTail << " columns";
#else
    GTEST_SKIP() << "needs a sparse anonymous mapping and a memory file";
#endif
}

// The group kernel reads the shift and the source scale of row m at m *
// shift_ld and m * src_scale_ld, the groups of K: at K 800000000 in groups
// of 4, row 11 of a 12-row tile lies past an int, in each order that reads
// them.  Both arrays sit in a sparse mapping in which only the rows' values
// are written; it reaches as far below them as an index wrapped to an int
// would, so such an index reads zeros instead of faulting.
TEST(NativeBrgemmInt4Bounds, RowStridesOfTwelveRowsPastAnInt) {
#if defined(__linux__)
    if (!vnni()) GTEST_SKIP() << "no AVX-512 VNNI";
    constexpr int kRows = 12;
    constexpr size_t kGroups = 800000000 / 4;
    ASSERT_GT((kRows - 1) * kGroups,
            static_cast<size_t>(std::numeric_limits<int>::max()));
    const size_t reach = (kRows - 1) * kGroups * sizeof(int32_t);
    struct mapping_t {
        void *at = MAP_FAILED;
        size_t bytes = 0;
        ~mapping_t() {
            if (at != MAP_FAILED) munmap(at, bytes);
        }
    } map;
    map.bytes = 2 * reach + 2 * sizeof(int32_t);
    map.at = mmap(nullptr, map.bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (map.at == MAP_FAILED)
        GTEST_SKIP() << "no address space for the sparse rows";
    // The shift of row m, then its source scale, at word m * kGroups.
    char *const base = static_cast<char *>(map.at) + reach;
    int32_t *const shift = reinterpret_cast<int32_t *>(base);
    float *const src_scale = reinterpret_cast<float *>(base + sizeof(int32_t));
    // Row m: four source values m + 1 over one K quad of weights 1 (nibble
    // 9 = w + 8), so the group's sum is 4 (m + 1); with source scale (m + 1)
    // / 2 and weight scale 1 every order gives 2 (m + 1)^2, exactly.
    std::vector<int8_t> a(static_cast<size_t>(kRows) * 8);
    for (int m = 0; m < kRows; ++m) {
        std::fill_n(a.begin() + m * 8, 8, static_cast<int8_t>(m + 1));
        shift[m * kGroups] = -8 * 4 * (m + 1);
        src_scale[m * kGroups] = 0.5f * static_cast<float>(m + 1);
    }
    const std::vector<uint8_t> b(64, 0x09);
    const std::vector<float> wei_scale(16, 1.0f);
    struct kind_t {
        nat::native_int8_order_t order;
        bool src_groups;
        bool finish;
    };
    for (const kind_t kind : {
                 kind_t {nat::native_int8_order_t::fused, true, false},
                 kind_t {nat::native_int8_order_t::fused, false, true},
                 kind_t {nat::native_int8_order_t::reference, false, false}}) {
        SCOPED_TRACE(std::string(kind.src_groups ? "source groups"
                        : kind.finish            ? "fused"
                                                 : "reference"));
        std::vector<float> c(static_cast<size_t>(kRows) * 16, -1.0f);
        nat::native_int4_group_ukernel_args_t args;
        args.A = reinterpret_cast<const uint8_t *>(a.data());
        args.lda = 8;
        args.a_step = 4;
        args.B = b.data();
        args.b_stride = 64;
        args.quad0 = 0;
        args.k_quads = 1;
        args.batch = 1;
        args.shift = shift;
        args.shift_ld = kGroups;
        args.shift_step = 1;
        args.src_scale = src_scale;
        args.src_scale_ld = kGroups;
        args.src_scale_step = kind.src_groups ? 1 : 0;
        args.wei_scale = wei_scale.data();
        args.wei_scale_step = 0;
        args.wei_scale_dt = kF32;
        args.C = c.data();
        args.ldc = 16;
        args.finish = kind.finish;
        const auto kernel = nat::select_native_int4_group_ukernel(
                kRows, 1, kind.order, kind.src_groups);
        ASSERT_NE(kernel, nullptr);
        kernel(args);
        for (int m = 0; m < kRows; ++m)
            for (int n = 0; n < 16; ++n)
                EXPECT_EQ(c[static_cast<size_t>(m) * 16 + n],
                        2.0f * static_cast<float>((m + 1) * (m + 1)))
                        << "row " << m << " column " << n;
    }
#else
    GTEST_SKIP() << "needs a sparse anonymous mapping";
#endif
}

} // namespace
