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

// The native layout of a weight depends on the weight alone: a gated call
// reads the layout a plain call on the same weight left (cached, or written
// in place over the weight's buffer), and the two share it.  Under the
// process-wide ZENDNNL_MATMUL_ALGO=10, for a BF16 weight, an s8 weight of a
// per-token quantized bf16 source (DA8W8) and an s4 one (DA8W4): a plain
// call first, then silu_and_mul, gelu_and_mul and swiglu_oai_mul calls on
// the same buffer, at weight cache modes 1 and 2, for N whose halves pair
// up in the tiles and for N that stage their rows.  Every gated call is
// served with the bits of the same call packed for itself from a pristine
// copy of the weight, and the weight keeps one layout throughout.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#if defined(__linux__)
#include <unistd.h>
#include <sys/mman.h>
#endif

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_view.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace nt = native_brgemm_test;
namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
using nt::buffer_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;
using gated_t = mm::matmul_gated_act_t;

constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kS4 = data_type_t::s4;
constexpr int kGroup = 32;

bool cpu_ok(data_type_t wei) {
    const auto &u = nat::detect_uarch();
    if (!nat::native_hw().base_isa) return false;
    return wei == kBf16 ? u.avx512bf16 : u.avx512vnni;
}

const char *wei_name(data_type_t wei) {
    return wei == kBf16 ? "bf16" : (wei == kS8 ? "s8" : "s4");
}

/// The operands of a case: a bf16 source, a raw [N, K] weight (bf16, s8,
/// or s4 two per byte) and, for an integer weight, f32 weight scales per
/// column (s8) or per group of kGroup along K (s4).
struct operands_t {
    buffer_t src, raw, wei_scale;
};

size_t weight_bytes(data_type_t wei, int K, int N) {
    const size_t n = static_cast<size_t>(K) * N;
    return wei == kS4 ? n / 2 : n * nt::dt_bytes(wei);
}

operands_t make_operands(data_type_t wei, int M, int N, int K, uint32_t seed) {
    operands_t o;
    o.src = nt::random_buffer(
            kBf16, static_cast<size_t>(M) * K, seed, -1.0f, 1.0f);
    if (wei == kBf16)
        o.raw = nt::random_buffer(
                kBf16, static_cast<size_t>(K) * N, seed + 1, -0.5f, 0.5f);
    else
        o.raw = nt::random_buffer(data_type_t::u8, weight_bytes(wei, K, N),
                seed + 1, 0.0f, 255.0f);
    if (wei == kS8)
        o.wei_scale = nt::random_buffer(kF32, N, seed + 2, 0.002f, 0.01f);
    else if (wei == kS4)
        o.wei_scale = nt::random_buffer(kF32,
                static_cast<size_t>(K / kGroup) * N, seed + 2, 0.01f, 0.05f);
    return o;
}

buffer_t copy_of(const buffer_t &b) {
    buffer_t c = nt::make_buffer(b.bytes);
    std::memcpy(c.get(), b.get(), b.bytes);
    return c;
}

/// The call on weight wei into dst; a source scale buffer per call, since a
/// dynamic quantization writes its scales there.
nt::call_t make_call(data_type_t wei_dt, int M, int N, int K, int wc,
        gated_t gated, const operands_t &o, const void *wei, void *dst,
        buffer_t &src_scale) {
    nt::call_t c;
    c.M = M;
    c.N = N;
    c.K = K;
    c.transB = true;
    c.src = o.src.get();
    c.wei = wei;
    c.dst = dst;
    c.gated = gated;
    auto &p = c.params;
    p.dtypes.src = kBf16;
    p.dtypes.wei = wei_dt;
    p.dtypes.dst = kBf16;
    p.weight_cache_type = wc;
    if (wei_dt != kBf16) {
        p.dynamic_quant = true;
        p.dtypes.compute = kS8;
        src_scale = nt::make_buffer(static_cast<size_t>(M) * 4);
        auto &qp = p.quant_params;
        qp.src_scale.buff = src_scale.get();
        qp.src_scale.dt = kF32;
        qp.src_scale.dims = {M, 1};
        qp.wei_scale.buff = o.wei_scale.get();
        qp.wei_scale.dt = kF32;
        if (wei_dt == kS8)
            qp.wei_scale.dims = {1, N};
        else
            qp.wei_scale.dims = {K / kGroup, N};
    }
    return c;
}

int out_cols(int N, gated_t gated) {
    return gated == gated_t::none ? N : N / 2;
}

/// The call on a pristine copy of the raw weight, packed for this call
/// only (weight cache mode 0): the bits a gated call must give.
buffer_t fresh_result(data_type_t wei_dt, int M, int N, int K, gated_t gated,
        const operands_t &o) {
    const buffer_t wei = nt::kept_until_exit(copy_of(o.raw));
    buffer_t dst
            = nt::make_buffer(static_cast<size_t>(M) * out_cols(N, gated) * 2);
    buffer_t src_scale;
    const nt::call_t c = make_call(
            wei_dt, M, N, K, 0, gated, o, wei.get(), dst.get(), src_scale);
    EXPECT_EQ(nt::run_matmul(c), status_t::success);
    return dst;
}

TEST(NativeGatedPack, PlainLayoutServesLaterGatedCalls) {
    const int M = 5, K = 128;
    uint32_t seed = 4100;
    for (data_type_t wei_dt : {kBf16, kS8, kS4}) {
        if (!cpu_ok(wei_dt)) continue;
        for (int wc : {1, 2})
            for (int N : {64, 96, 40}) {
                const std::string name = std::string(wei_name(wei_dt)) + " wc="
                        + std::to_string(wc) + " N=" + std::to_string(N);
                SCOPED_TRACE(name);
                const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
                const operands_t o = make_operands(wei_dt, M, N, K, seed++);
                // The buffer the calls share; a weight cache 2 call may
                // rewrite it in place.
                const buffer_t wei = nt::kept_until_exit(copy_of(o.raw));
                {
                    buffer_t dst
                            = nt::make_buffer(static_cast<size_t>(M) * N * 2);
                    buffer_t src_scale;
                    const nt::call_t plain = make_call(wei_dt, M, N, K, wc,
                            gated_t::none, o, wei.get(), dst.get(), src_scale);
                    ASSERT_EQ(nt::run_matmul(plain), status_t::success);
                    EXPECT_EQ(nt::native_layouts(wei.get()), 1u);
                }
                for (gated_t g : {gated_t::silu_and_mul, gated_t::gelu_and_mul,
                             gated_t::gelu_tanh_and_mul,
                             gated_t::swiglu_oai_mul}) {
                    SCOPED_TRACE(
                            "gated " + std::to_string(static_cast<int>(g)));
                    const buffer_t want = fresh_result(wei_dt, M, N, K, g, o);
                    buffer_t dst = nt::make_buffer(want.bytes);
                    buffer_t src_scale;
                    const nt::call_t c = make_call(wei_dt, M, N, K, wc, g, o,
                            wei.get(), dst.get(), src_scale);
                    ASSERT_EQ(nt::run_matmul(c), status_t::success);
                    EXPECT_EQ(
                            std::memcmp(dst.get(), want.get(), want.bytes), 0);
                    EXPECT_EQ(nt::native_layouts(wei.get()), 1u);
                }
                // A plain call after the gated ones reads the same layout.
                buffer_t dst = nt::make_buffer(static_cast<size_t>(M) * N * 2);
                buffer_t src_scale;
                const nt::call_t plain = make_call(wei_dt, M, N, K, wc,
                        gated_t::none, o, wei.get(), dst.get(), src_scale);
                ASSERT_EQ(nt::run_matmul(plain), status_t::success);
                const buffer_t want
                        = fresh_result(wei_dt, M, N, K, gated_t::none, o);
                EXPECT_EQ(std::memcmp(dst.get(), want.get(), want.bytes), 0);
                EXPECT_EQ(nt::native_layouts(wei.get()), 1u);
                mm::clear_matmul_weight_caches();
            }
    }
}

// The gated results of BF16 calls over weights of every N tail, of [gate |
// up] halves and of interleaved pairs, against the scalar model: the
// reference algorithm's f32 pre-activation, then the activation in f32.
TEST(NativeGatedPack, Bf16GatedEveryNTailMatchesModel) {
    if (!cpu_ok(kBf16)) GTEST_SKIP() << "needs AVX512-BF16";
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int M = 6, K = 96;
    uint32_t seed = 7300;
    for (int N = 2; N <= 130; N += 2)
        for (gated_t g : {gated_t::silu_and_mul, gated_t::gelu_and_mul,
                     gated_t::gelu_tanh_and_mul, gated_t::swiglu_oai_mul}) {
            SCOPED_TRACE("N=" + std::to_string(N) + " gated "
                    + std::to_string(static_cast<int>(g)));
            const operands_t o = make_operands(kBf16, M, N, K, seed++);
            const int I = N / 2;
            buffer_t dst = nt::make_buffer(static_cast<size_t>(M) * I * 2);
            buffer_t src_scale;
            const nt::call_t c = make_call(
                    kBf16, M, N, K, 0, g, o, o.raw.get(), dst.get(), src_scale);
            ASSERT_EQ(nt::run_matmul(c), status_t::success);
            std::vector<float> pre(static_cast<size_t>(M) * N);
            ASSERT_EQ(nt::run_reference(c, o.raw.get(), pre.data()),
                    status_t::success);
            std::vector<float> want(static_cast<size_t>(M) * I);
            nt::gated_act_ref(pre.data(), M, N, g, want.data());
            int bad = 0;
            for (size_t i = 0; i < want.size(); ++i) {
                const float got = nt::load_f32(dst.get(), kBf16, i);
                if (std::fabs(got - want[i])
                        > 2e-3f + std::fabs(want[i]) * (1.0f / 128.0f))
                    ++bad;
            }
            EXPECT_EQ(bad, 0);
        }
}

// The width at which a gated call pairs its halves in its tiles: tiles of
// 64 columns or more as planned; narrower ones in whole blocks at 64
// columns for N a multiple of 64 whose tiles of 64 deal out evenly over the
// team (or number 8 per thread), else in runs of 16 columns in tiles of 32,
// for N a multiple of 32; 0 (staged rows) otherwise.  bf16 and INT8 tiles
// alike.
TEST(NativeGatedPack, HalvesPairAtTheWidthThePlanDealsOut) {
    struct case_t {
        int N, nr, m_blocks, threads;
        int width;
    };
    const case_t cases[] = {{3072, 32, 1, 32, 32}, {5120, 32, 1, 32, 32},
            {1024, 32, 1, 32, 32}, {1024, 32, 2, 32, 64},
            {12288, 32, 1, 32, 64}, {3072, 32, 4, 32, 64},
            {5120, 32, 4, 32, 64}, {4128, 32, 1, 32, 32},
            {11040, 32, 8, 32, 32}, {512, 16, 1, 32, 32}, {768, 16, 4, 32, 32},
            {3072, 32, 16, 32, 64}, {4100, 32, 1, 32, 0}, {11000, 32, 4, 32, 0},
            {12288, 128, 1, 32, 128}, {4096, 64, 1, 32, 64},
            {22016, 32, 1, 64, 32}, {28672, 32, 1, 64, 64}};
    for (const case_t &c : cases) {
        SCOPED_TRACE(::testing::Message()
                << "N " << c.N << " nr " << c.nr << " row blocks " << c.m_blocks
                << " threads " << c.threads);
        EXPECT_EQ(nat::native_paired_width(c.N, c.nr, c.m_blocks, c.threads),
                c.width);
    }
}

#if defined(__linux__)
// A gated call whose tiles pair runs of 16 columns (N 131104 or 65568, a
// multiple of 32 and not of 64) reads the up run of its first tile N / 2
// columns after the gate run: beyond the int range for a large weight,
// 4294967360 bytes (2147483680 bf16 elements) at K 32768 and N 131104 or K
// 65536 and N 65568 for bf16, 2147745856 bytes at K 32768 for s8 and
// 2147483712 bytes at K 65536 for s4.  The tiles take that
// distance whole, and the 12 paired bf16 kernels read their second 16
// columns there: over a sparse mapping that holds the first K pair of both
// runs and reads zero elsewhere, before the gate run too, each row's 32
// sums are those of its gate and up columns.
TEST(NativeGatedPack, UpRunsPastTheIntRangeAreReadWhereTheyLie) {
    const int N = 131104;
    ASSERT_EQ(nat::native_paired_width(N, 32, 1, 32), 32);
    ASSERT_EQ(nat::native_paired_width(65568, 32, 1, 32), 32);
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t up_bytes = 4294967360u;
    const size_t lead = (size_t {1} << 32) + page;
    const size_t bytes = lead + up_bytes + page;
    void *const map = mmap(nullptr, bytes, PROT_READ,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (map == MAP_FAILED) GTEST_SKIP() << "no address space for the weight";
    struct unmap_t {
        void *p;
        size_t n;
        ~unmap_t() { munmap(p, n); }
    } unmap {map, bytes};
    uint8_t *const base = static_cast<uint8_t *>(map) + lead;

    const auto view = [&](data_type_t wei_dt, int K, int cols) {
        nat::native_weight_view_t w;
        w.base = base;
        w.wei_dt = wei_dt;
        w.K = K;
        w.N = cols;
        w.pack_nr = nat::kNativeLayoutPackNr;
        w.k_groups = nat::native_k_groups(wei_dt, K);
        w.block_bytes = nat::native_block_bytes(wei_dt, K, w.pack_nr);
        w.paired = true;
        return w;
    };
    struct case_t {
        data_type_t wei_dt;
        int K, N;
        size_t distance;
    };
    for (const case_t &c : {case_t {kBf16, 32768, N, up_bytes},
                 case_t {kBf16, 65536, 65568, up_bytes},
                 case_t {kS8, 32768, N, 2147745856u},
                 case_t {kS4, 65536, N, 2147483712u}}) {
        SCOPED_TRACE(::testing::Message()
                << wei_name(c.wei_dt) << " K " << c.K << " N " << c.N);
        const nat::native_weight_view_t w = view(c.wei_dt, c.K, c.N);
        EXPECT_EQ(
                w.column_distance(w.tile_col(0, 32, 0), w.tile_col(0, 32, 16)),
                c.distance);
    }

    if (!cpu_ok(kBf16)) return;
    const nat::native_weight_view_t w = view(kBf16, 32768, N);
    const int up = w.tile_col(0, 32, 16);
    ASSERT_EQ(w.column(up), base + up_bytes);
    uint8_t *const up_page = base + up_bytes / page * page;
    ASSERT_EQ(mprotect(base, page, PROT_READ | PROT_WRITE), 0);
    ASSERT_EQ(mprotect(up_page, page, PROT_READ | PROT_WRITE), 0);
    const auto bf16 = [](float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        return static_cast<uint16_t>(bits >> 16);
    };
    // K pair 0 of 16 columns, two elements each: gate g(j, e) = j + 1 + e,
    // up u(j, e) = -(j + 2 + 3 e).
    uint16_t *const gate_run = reinterpret_cast<uint16_t *>(base);
    uint16_t *const up_run = reinterpret_cast<uint16_t *>(base + up_bytes);
    for (int j = 0; j < 16; ++j)
        for (int e = 0; e < 2; ++e) {
            gate_run[2 * j + e] = bf16(static_cast<float>(j + 1 + e));
            up_run[2 * j + e] = bf16(-static_cast<float>(j + 2 + 3 * e));
        }
    const nat::bf16_brgemm_paired_fn_t *paired
            = nat::bf16_brgemm_paired_table();
    for (int mr = 1; mr <= 12; ++mr) {
        SCOPED_TRACE(::testing::Message() << "MR " << mr);
        uint16_t a[12 * 2];
        for (int m = 0; m < mr; ++m)
            for (int e = 0; e < 2; ++e)
                a[2 * m + e] = bf16(static_cast<float>(m + 1 + e));
        std::vector<float> c(static_cast<size_t>(mr) * 32, -1.0f);
        paired[mr](a, 2, gate_run, w.group_stride(),
                w.column_distance(0, up) / 2, c.data(), 32, 2, 2, 0.0f, nullptr,
                nat::fused_postop_t::none, nullptr, 0);
        int bad = 0;
        for (int m = 0; m < mr; ++m)
            for (int j = 0; j < 16; ++j) {
                const float a0 = m + 1.0f, a1 = m + 2.0f;
                const float g = a0 * (j + 1) + a1 * (j + 2);
                const float u = -(a0 * (j + 2) + a1 * (j + 5));
                bad += c[m * 32 + j] != g;
                bad += c[m * 32 + 16 + j] != u;
            }
        EXPECT_EQ(bad, 0);
    }
}
#endif

// Gated calls over bf16, s8 and s4 weights whose halves pair in whole
// blocks (N 12288), in runs of 16 columns (N 1024, 3072 and 4128 at a few
// rows on 32 threads, N 512 of 16-column tiles) or stage their rows (N
// 4100): each gives the activation of the same call's f32 pre-activation
// without the gated activation, within the rounding of its bf16 dst.
TEST(NativeGatedPack, PairedRunsOfEveryWidthMatchThePlainCall) {
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int K = 128;
    struct shape_t {
        int M, N;
    };
    uint32_t seed = 7500;
    for (const data_type_t wei_dt : {kBf16, kS8, kS4}) {
        if (!cpu_ok(wei_dt)) continue;
        for (const shape_t sh : {shape_t {1, 1024}, shape_t {16, 1024},
                     shape_t {4, 3072}, shape_t {16, 4128}, shape_t {16, 512},
                     shape_t {1, 12288}, shape_t {16, 4100}}) {
            SCOPED_TRACE(::testing::Message()
                    << wei_name(wei_dt) << " M " << sh.M << " N " << sh.N);
            const operands_t o = make_operands(wei_dt, sh.M, sh.N, K, seed++);
            const int I = sh.N / 2;
            buffer_t pre
                    = nt::make_buffer(static_cast<size_t>(sh.M) * sh.N * 4);
            buffer_t plain_scale, gated_scale;
            nt::call_t plain = make_call(wei_dt, sh.M, sh.N, K, 0,
                    gated_t::none, o, o.raw.get(), pre.get(), plain_scale);
            plain.params.dtypes.dst = kF32;
            plain.params.num_threads = 32;
            ASSERT_EQ(nt::run_matmul(plain), status_t::success);
            std::vector<float> want(static_cast<size_t>(sh.M) * I);
            nt::gated_act_ref(pre.as<float>(), sh.M, sh.N,
                    gated_t::silu_and_mul, want.data());
            buffer_t dst = nt::make_buffer(static_cast<size_t>(sh.M) * I * 2);
            nt::call_t gated
                    = make_call(wei_dt, sh.M, sh.N, K, 0, gated_t::silu_and_mul,
                            o, o.raw.get(), dst.get(), gated_scale);
            gated.params.num_threads = 32;
            ASSERT_EQ(nt::run_matmul(gated), status_t::success);
            int bad = 0;
            for (size_t i = 0; i < want.size(); ++i) {
                const float got = nt::load_f32(dst.get(), kBf16, i);
                if (std::fabs(got - want[i])
                        > 1e-3f + std::fabs(want[i]) * (1.0f / 128.0f))
                    ++bad;
            }
            EXPECT_EQ(bad, 0);
        }
    }
}

// A gated call over [gate | up] halves names on its plan line the width at
// which its tiles paired the halves, or that it staged its rows, and runs
// what it names: N 4128 and 11040 (halves of runs of 16 columns inside the
// blocks) pair at 32 columns, N 1024 at a few rows on 32 threads at 32 for
// s8 and s4 (16 tiles of 64 columns would idle half the team), N 12288 of
// INT8 tiles at 64, and N 4100 stages its rows.  Each output matches the
// activation of the same call's f32 pre-activation.
TEST(NativeGatedPack, PlanLineNamesThePairingTheCallRan) {
    nt::expect_in_child({{"ZENDNNL_API_LOG_LEVEL", "4"},
                                {"ZENDNNL_MATMUL_NATIVE_CONFIG", nullptr},
                                {"ZENDNNL_MATMUL_NATIVE_TUNING_FILE", nullptr}},
            [] {
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        struct case_t {
            int M, N;
            const char *int8, *bf16;
        };
        // nullptr: any width, not staged (the planner's bf16 tile width).
        const case_t cases[] = {{16, 4128, "32", "32"}, {4, 11040, "32", "32"},
                {16, 1024, "32", nullptr}, {1, 12288, "64", nullptr},
                {16, 4100, "staged", "staged"}};
        const int K = 128;
        uint32_t seed = 7700;
        int bad = 0;
        for (const data_type_t wei_dt : {kBf16, kS8, kS4}) {
            if (!cpu_ok(wei_dt)) continue;
            for (const case_t &cs : cases) {
                const operands_t o
                        = make_operands(wei_dt, cs.M, cs.N, K, seed++);
                const int I = cs.N / 2;
                buffer_t pre
                        = nt::make_buffer(static_cast<size_t>(cs.M) * cs.N * 4);
                buffer_t plain_scale, gated_scale;
                nt::call_t plain = make_call(wei_dt, cs.M, cs.N, K, 0,
                        gated_t::none, o, o.raw.get(), pre.get(), plain_scale);
                plain.params.dtypes.dst = kF32;
                plain.params.num_threads = 32;
                buffer_t dst
                        = nt::make_buffer(static_cast<size_t>(cs.M) * I * 2);
                nt::call_t gated = make_call(wei_dt, cs.M, cs.N, K, 0,
                        gated_t::silu_and_mul, o, o.raw.get(), dst.get(),
                        gated_scale);
                gated.params.num_threads = 32;
                ::testing::internal::CaptureStdout();
                const status_t pst = nt::run_matmul(plain);
                const status_t gst = nt::run_matmul(gated);
                const std::string log
                        = ::testing::internal::GetCapturedStdout();
                // The gated call's line, the last of the two calls'.
                const std::string head
                        = "[MATMUL.NATIVE_BRGEMM] M=" + std::to_string(cs.M)
                        + " N=" + std::to_string(cs.N) + " ";
                const size_t at = log.rfind(head);
                const std::string line = at == std::string::npos
                        ? std::string()
                        : log.substr(at, log.find('\n', at) - at);
                const size_t pf = line.find(" pairing=");
                const std::string pairing = pf == std::string::npos
                        ? std::string()
                        : line.substr(pf + 9, line.find(' ', pf + 9) - pf - 9);
                const char *want = wei_dt == kBf16 ? cs.bf16 : cs.int8;
                const size_t nr_at = line.find(" nr=");
                const std::string nr = nr_at == std::string::npos
                        ? std::string()
                        : line.substr(nr_at + 4,
                                  line.find(' ', nr_at + 4) - nr_at - 4);
                bool ok = pst == status_t::success && gst == status_t::success
                        && !pairing.empty()
                        && (want != nullptr ? pairing == want
                                            : pairing != "staged")
                        && (pairing == "staged" || pairing == nr);
                std::vector<float> ref(static_cast<size_t>(cs.M) * I);
                nt::gated_act_ref(pre.as<float>(), cs.M, cs.N,
                        gated_t::silu_and_mul, ref.data());
                int off = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    const float got = nt::load_f32(dst.get(), kBf16, i);
                    if (std::fabs(got - ref[i])
                            > 1e-3f + std::fabs(ref[i]) * (1.0f / 128.0f))
                        ++off;
                }
                if (!ok || off != 0) {
                    std::fprintf(stderr,
                            "%s M=%d N=%d: pairing '%s' (want %s), %d outputs "
                            "off, line: %s\n",
                            wei_name(wei_dt), cs.M, cs.N, pairing.c_str(),
                            want != nullptr ? want : "paired", off,
                            line.c_str());
                    ++bad;
                }
            }
        }
        return bad;
    }, "the pairing of gated calls");
}

// gelu_tanh_and_mul, the tanh form of gelu_and_mul, runs on the native
// BRGEMM only: with the algorithm unset the call returns unimplemented and
// leaves dst as it was; with the native BRGEMM asked for, per call or for
// the process, it is served and matches the scalar model.
TEST(NativeGatedPack, GeluTanhOnlyOnTheNativeBrgemm) {
    if (!cpu_ok(kBf16)) GTEST_SKIP() << "needs AVX512-BF16";
    const int M = 3, N = 64, K = 64;
    const operands_t o = make_operands(kBf16, M, N, K, 9100);
    const size_t out = static_cast<size_t>(M) * (N / 2);
    std::vector<float> pre(static_cast<size_t>(M) * N);
    std::vector<float> want(out);
    {
        buffer_t dst = nt::make_buffer(out * 2);
        buffer_t src_scale;
        const nt::call_t c
                = make_call(kBf16, M, N, K, 0, gated_t::gelu_tanh_and_mul, o,
                        o.raw.get(), dst.get(), src_scale);
        ASSERT_EQ(nt::run_reference(c, o.raw.get(), pre.data()),
                status_t::success);
        nt::gated_act_ref(
                pre.data(), M, N, gated_t::gelu_tanh_and_mul, want.data());
    }
    for (int route = 0; route < 3; ++route) {
        SCOPED_TRACE("route " + std::to_string(route));
        const nt::algo_pin_t pin(route == 2 ? matmul_algo_t::native_brgemm
                                            : matmul_algo_t::none);
        buffer_t dst = nt::make_buffer(out * 2);
        std::memset(dst.get(), 0x5a, out * 2);
        const buffer_t before = copy_of(dst);
        buffer_t src_scale;
        nt::call_t c = make_call(kBf16, M, N, K, 0, gated_t::gelu_tanh_and_mul,
                o, o.raw.get(), dst.get(), src_scale);
        if (route == 1) c.params.lowoha_algo = matmul_algo_t::native_brgemm;
        if (route == 0) {
            if (!nt::default_route_reachable()) continue;
            EXPECT_EQ(nt::run_matmul(c), status_t::unimplemented);
            EXPECT_EQ(std::memcmp(dst.get(), before.get(), out * 2), 0);
            continue;
        }
        ASSERT_EQ(nt::run_matmul(c), status_t::success);
        int bad = 0;
        for (size_t i = 0; i < out; ++i) {
            const float got = nt::load_f32(dst.get(), kBf16, i);
            if (std::fabs(got - want[i])
                    > 2e-3f + std::fabs(want[i]) * (1.0f / 128.0f))
                ++bad;
        }
        EXPECT_EQ(bad, 0);
    }
}

} // namespace
