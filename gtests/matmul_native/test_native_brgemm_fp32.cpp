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

// FP32 in the native BRGEMM (ALGO 10): the kernels of every row count and
// width over K tails against a scalar model, the f32 native layout, the
// plans of FP32 problems, the outputs bit for bit as the products added in K
// order, and the FP32 calls matmul_direct takes to the native BRGEMM and the
// weight cache keeps for it.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/fp32/native_fp32_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_view.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using nt::buffer_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr int kNr = nat::kNativeLayoutPackNr;

#define SKIP_WITHOUT_FP32_ISA() \
    do { \
        if (!nat::native_base_isa()) \
            GTEST_SKIP() << "needs AVX-512 F, BW, VL and DQ and FMA"; \
    } while (0)

std::vector<float> random_floats(
        size_t n, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    const buffer_t b = nt::random_buffer(kF32, n, seed, lo, hi);
    return std::vector<float>(b.as<float>(), b.as<float>() + n);
}

uint32_t bits_of(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

// ── Kernels ──────────────────────────────────────────────────────────────

/// The tile the kernel computes: beta * C rounded once (+0.0 when beta is
/// 0), then one fused multiply-add per k in K order.
void model_tile(const std::vector<float> &A, int lda,
        const std::vector<float> &B, int b_stride, std::vector<float> &C,
        int ldc, int mr, int nr, int K, float beta) {
    for (int m = 0; m < mr; ++m)
        for (int n = 0; n < nr; ++n) {
            float acc = beta != 0.0f ? beta * C[m * ldc + n] : 0.0f;
            for (int k = 0; k < K; ++k)
                acc = std::fma(A[static_cast<size_t>(m) * lda + k],
                        B[static_cast<size_t>(k) * b_stride + n], acc);
            C[m * ldc + n] = acc;
        }
}

// Every row count 1 .. 12 at both widths, over K of every residue of the
// four-step loop and K blocks that leave a tail, with beta 0 (C holds NaN,
// which must not be read), 1 and others: the model's bits, and nothing
// outside the tile written.
TEST(NativeBrgemmFp32Kernel, EveryRowCountWidthAndKTailMatchesTheModel) {
    SKIP_WITHOUT_FP32_ISA();
    const int Ks[] = {1, 2, 3, 4, 5, 7, 8, 13, 64, 67, 259};
    const float betas[] = {0.0f, 1.0f, 0.5f, -1.25f};
    int cases = 0, failures = 0;
    for (const int nv : {1, 2}) {
        const int nr = 16 * nv;
        for (const int b_stride : {kNr, nr + 24})
            for (int mr = 1; mr <= 12; ++mr) {
                const nat::native_fp32_ukernel_fn_t fn
                        = nat::select_native_fp32_ukernel(mr, nv);
                ASSERT_NE(fn, nullptr) << "mr " << mr << " nv " << nv;
                for (const int K : Ks)
                    for (const int BK : {K, 4, 5, 16})
                        for (const float beta : betas) {
                            const int lda = K + 3, ldc = nr + 5;
                            const uint32_t seed = static_cast<uint32_t>(
                                    mr * 1000 + K * 10 + nv);
                            const std::vector<float> A = random_floats(
                                    static_cast<size_t>(mr) * lda, seed);
                            const std::vector<float> B = random_floats(
                                    static_cast<size_t>(K) * b_stride,
                                    seed + 1);
                            std::vector<float> C0 = random_floats(
                                    static_cast<size_t>(mr) * ldc, seed + 2);
                            if (beta == 0.0f)
                                for (int m = 0; m < mr; ++m)
                                    for (int n = 0; n < nr; ++n)
                                        C0[m * ldc + n] = std::numeric_limits<
                                                float>::quiet_NaN();
                            std::vector<float> got = C0, want = C0;
                            fn(A.data(), lda, B.data(), b_stride, got.data(),
                                    ldc, K, BK, beta);
                            model_tile(A, lda, B, b_stride, want, ldc, mr, nr,
                                    K, beta);
                            ++cases;
                            bool ok = true;
                            for (size_t i = 0; i < got.size() && ok; ++i)
                                ok = bits_of(got[i]) == bits_of(want[i]);
                            if (!ok && failures++ < 10)
                                ADD_FAILURE()
                                        << "mr " << mr << " nr " << nr << " K "
                                        << K << " BK " << BK << " beta " << beta
                                        << " b_stride " << b_stride;
                        }
            }
    }
    EXPECT_EQ(failures, 0) << "of " << cases << " cases";
    for (const int nv : {0, 3})
        EXPECT_EQ(nat::select_native_fp32_ukernel(1, nv), nullptr);
    for (const int mr : {0, 13})
        EXPECT_EQ(nat::select_native_fp32_ukernel(mr, 1), nullptr);
}

// ── Layout ───────────────────────────────────────────────────────────────

// Each element of an f32 weight at its place in the layout: block p / 32,
// row k, column p % 32 for pack column p, from [K, N] or [N, K] storage,
// with a padded ldb, K past the tiled walk of a [K, N] weight, more blocks
// than one tile and more than one thread.
TEST(NativeBrgemmFp32Layout, PackPlacesEveryElement) {
    SKIP_WITHOUT_FP32_ISA();
    int checked = 0;
    for (const int K : {1, 3, 64, 1024, 1031})
        for (const int N : {32, 96, 544})
            for (const bool transB : {false, true})
                for (const int pad : {0, 5})
                    for (const int threads : {1, 4}) {
                        if (K == 1031 && N == 544 && threads == 1) continue;
                        SCOPED_TRACE("K=" + std::to_string(K)
                                + " N=" + std::to_string(N)
                                + " transB=" + std::to_string(transB)
                                + " pad=" + std::to_string(pad)
                                + " threads=" + std::to_string(threads));
                        const int ldb = (transB ? K : N) + pad;
                        std::vector<float> w(
                                static_cast<size_t>(transB ? N : K) * ldb);
                        for (size_t i = 0; i < w.size(); ++i)
                            w[i] = static_cast<float>(i % 100003);
                        const size_t bytes
                                = nat::native_packed_bytes(kF32, K, N, kNr);
                        ASSERT_EQ(bytes,
                                (static_cast<size_t>(K) * N * 4 + 63)
                                        & ~size_t {63});
                        ASSERT_EQ(nat::native_block_bytes(kF32, K, kNr),
                                static_cast<size_t>(K) * kNr * 4);
                        std::vector<float> packed(bytes / 4, -1.0f);
                        ASSERT_EQ(nat::native_pack_weight(kF32, w.data(), K, N,
                                          ldb, kNr, transB, packed.data(),
                                          threads),
                                status_t::success);
                        int bad = 0;
                        for (int p = 0; p < N; ++p) {
                            for (int k = 0; k < K; ++k) {
                                const size_t from = transB
                                        ? static_cast<size_t>(p) * ldb + k
                                        : static_cast<size_t>(k) * ldb + p;
                                const size_t at
                                        = static_cast<size_t>(p / kNr) * K * kNr
                                        + static_cast<size_t>(k) * kNr
                                        + p % kNr;
                                bad += packed[at] != w[from];
                            }
                        }
                        EXPECT_EQ(bad, 0);
                        ++checked;
                    }
    EXPECT_GT(checked, 0);
}

// The layout's facts of an f32 weight: K rows per block, a group stride of
// one block row, and the layout rule's shape (any N, held in whole blocks).
TEST(NativeBrgemmFp32Layout, ViewAndRuleFacts) {
    EXPECT_EQ(nat::native_k_groups(kF32, 1031), 1031);
    EXPECT_TRUE(nat::native_layout_fits(kF32, 1, 32, kNr));
    EXPECT_TRUE(nat::native_layout_fits(kF32, 64, 48, kNr));
    EXPECT_EQ(nat::native_packed_bytes(kF32, 64, 48, kNr),
            static_cast<size_t>(64) * 64 * 4);
    EXPECT_FALSE(nat::native_layout_fits(kF32, 0, 64, kNr));
    EXPECT_FALSE(nat::native_layout_fits(kF32, 64, 0, kNr));
    nat::native_weight_view_t v;
    v.wei_dt = kF32;
    v.pack_nr = kNr;
    EXPECT_EQ(v.group_stride(), kNr);
}

// ── Plans ────────────────────────────────────────────────────────────────

nat::native_problem_t problem(int M, int N, int K, int threads, data_type_t wei,
        data_type_t dst = kF32) {
    nat::native_problem_t p;
    p.M = M;
    p.N = N;
    p.K = K;
    p.pack_nr = kNr;
    p.threads = threads;
    p.src_dt = wei;
    p.wei_dt = wei;
    p.dst_dt = dst;
    return p;
}

nat::heuristic_params_t heuristic_of(const char *text) {
    nat::native_config_t cfg;
    std::string why;
    EXPECT_TRUE(nat::parse_native_config(text, cfg, why))
            << text << ": " << why;
    return nat::apply_native_config({}, cfg);
}

// The L2 rule weighs an FP32 B slab at four bytes per element: the K pass is
// the largest even K whose slab fits the bf16_l2_pct share of L2, at least
// min_pass_k, as for BF16 at two bytes.
TEST(NativeBrgemmFp32Plan, KPassWeighsFourBytesPerElement) {
    SKIP_WITHOUT_FP32_ISA();
    const nat::UarchParams &u = nat::native_plan_uarch();
    const nat::heuristic_params_t hp = heuristic_of(
            "makespan=0,l1_pass=0,traffic_groups=0,bf16_alias=0,"
            "bf16_narrow_n=0,blocks=0,nr=32,mr=12,mb=96,threads_cap=1");
    for (const int K : {512, 4096, 9000, 65536}) {
        const nat::native_problem_t p = problem(96, 256, K, 1, kF32);
        const nat::native_plan_t q = nat::heuristic_plan(p, u, hp);
        ASSERT_EQ(nat::validate_native_plan(q, p), nullptr) << K;
        const int slab = static_cast<int>(static_cast<int64_t>(u.l2_bytes)
                * hp.bf16_l2_pct / 100 / (32 * 4));
        const int want = slab < K ? std::max(hp.min_pass_k, slab & ~1) : K;
        EXPECT_EQ(q.kc, want) << "K " << K;
        const nat::native_plan_t b
                = nat::heuristic_plan(problem(96, 256, K, 1, kBf16), u, hp);
        const int bf16_slab = static_cast<int>(static_cast<int64_t>(u.l2_bytes)
                * hp.bf16_l2_pct / 100 / (32 * 2));
        EXPECT_EQ(b.kc,
                bf16_slab < K ? std::max(hp.min_pass_k, bf16_slab & ~1) : K)
                << "K " << K;
        // K passes keep the accumulators of the row block.
        EXPECT_EQ(q.scratch_floats,
                q.kc < K ? static_cast<size_t>(q.mb) * q.nr
                         : static_cast<size_t>(q.mr) * q.nr);
    }
}

// FP32 rows per call follow the BF16 rule (tall_rows from 32 columns), and
// the alias rule applies at K a multiple of 1024, where FP32 source rows are
// a multiple of 4 KiB apart.
TEST(NativeBrgemmFp32Plan, RowsPerCallAndTheAliasRule) {
    SKIP_WITHOUT_FP32_ISA();
    const nat::UarchParams &u = nat::native_plan_uarch();
    const nat::heuristic_params_t hp = heuristic_of(
            "makespan=0,l1_pass=0,traffic_groups=0,bf16_narrow_n=0,blocks=0,"
            "nr=32,mb=192,threads_cap=1");
    for (const int K : {1024, 2048, 3072, 512, 1000}) {
        const nat::native_problem_t p = problem(1024, 256, K, 1, kF32);
        const nat::native_plan_t q = nat::heuristic_plan(p, u, hp);
        ASSERT_EQ(nat::validate_native_plan(q, p), nullptr) << K;
        EXPECT_EQ(q.mr, K % 1024 == 0 ? hp.bf16_alias_mr : hp.tall_rows)
                << "K " << K;
    }
    const nat::heuristic_params_t narrow = heuristic_of("nr=16,threads_cap=1");
    const nat::native_problem_t p = problem(64, 256, 64, 1, kF32);
    EXPECT_EQ(nat::heuristic_plan(p, u, narrow).mr, narrow.tile_rows);
}

// An FP32 plan takes tiles of 16 or 32 columns and of 64 and 128 columns of
// 32-column slabs, any K pass of at least one and the FP32 kernels' rows;
// odd K passes stay BF16's.
TEST(NativeBrgemmFp32Plan, ValidateTheFp32Tile) {
    SKIP_WITHOUT_FP32_ISA();
    const nat::UarchParams &u = nat::native_plan_uarch();
    const nat::native_problem_t f = problem(64, 256, 1000, 4, kF32);
    const nat::native_problem_t b = problem(64, 256, 1000, 4, kBf16);
    for (const char *text : {"nr=16", "nr=32", "kc=3", "kc=999", "mr=12",
                 "nr=16,mr=12", "schedule=blocks,unit_tiles=3,kc=64", "nr=64",
                 "nr=128", "nr=128,mr=7,kc=3"}) {
        const nat::native_plan_t q
                = nat::heuristic_plan(f, u, heuristic_of(text));
        EXPECT_EQ(nat::validate_native_plan(q, f), nullptr) << text;
    }
    nat::native_plan_t q = nat::heuristic_plan(
            problem(64, 288, 1000, 4, kF32), u, heuristic_of("nr=64"));
    EXPECT_NE(nat::validate_native_plan(q, problem(64, 288, 1000, 4, kF32)),
            nullptr);
    if (nat::detect_uarch().avx512bf16) {
        q = nat::heuristic_plan(b, u, heuristic_of("kc=3"));
        EXPECT_NE(nat::validate_native_plan(q, b), nullptr);
        q = nat::heuristic_plan(b, u, heuristic_of("nr=64"));
        EXPECT_EQ(nat::validate_native_plan(q, b), nullptr);
    }
    q = nat::heuristic_plan(f, u, {});
    q.kc = 0;
    q.scratch_floats = nat::native_plan_scratch_floats(q, f);
    EXPECT_NE(nat::validate_native_plan(q, f), nullptr);
}

// ── Outputs ──────────────────────────────────────────────────────────────

/// One FP32 call: dst = alpha * src * wei (+ bias), then relu; dst rows
/// N + ldc_pad apart.
struct fp32_call_t {
    int M = 0, N = 0, K = 0;
    bool transB = false;
    float alpha = 1.0f;
    bool bias = false;
    bool relu = false;
    int threads = 1;
    int ldc_pad = 0;

    int ldc() const { return N + ldc_pad; }
};

/// The value a run leaves in the dst elements it must not write.
constexpr float kUnwritten = -7.0f;

std::string str(const fp32_call_t &c) {
    return "M=" + std::to_string(c.M) + " N=" + std::to_string(c.N) + " K="
            + std::to_string(c.K) + " transB=" + std::to_string(c.transB)
            + " alpha=" + std::to_string(c.alpha) + " bias="
            + std::to_string(c.bias) + " relu=" + std::to_string(c.relu)
            + " threads=" + std::to_string(c.threads)
            + " ldc=" + std::to_string(c.ldc());
}

struct fp32_operands_t {
    std::vector<float> src, wei, bias;
    int lda = 0, ldb = 0;
};

fp32_operands_t operands_of(const fp32_call_t &c, uint32_t seed) {
    fp32_operands_t o;
    o.lda = c.K + 3;
    o.ldb = (c.transB ? c.K : c.N) + 5;
    o.src = random_floats(static_cast<size_t>(c.M) * o.lda, seed);
    o.wei = random_floats(
            static_cast<size_t>(c.transB ? c.N : c.K) * o.ldb, seed + 1);
    if (c.bias) o.bias = random_floats(c.N, seed + 2);
    return o;
}

/// The products added in K order from +0.0, one fused multiply-add each;
/// then alpha (one product), the bias (one sum) and relu.  Rows ldc apart,
/// kUnwritten past N.
std::vector<float> model_output(
        const fp32_call_t &c, const fp32_operands_t &o) {
    std::vector<float> out(static_cast<size_t>(c.M) * c.ldc(), kUnwritten);
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < c.K; ++k) {
                const float w = c.transB
                        ? o.wei[static_cast<size_t>(n) * o.ldb + k]
                        : o.wei[static_cast<size_t>(k) * o.ldb + n];
                acc = std::fma(
                        o.src[static_cast<size_t>(m) * o.lda + k], w, acc);
            }
            if (c.alpha != 1.0f) acc = acc * c.alpha;
            if (c.bias) acc = acc + o.bias[n];
            if (c.relu) acc = acc > 0.0f ? acc : 0.0f;
            out[static_cast<size_t>(m) * c.ldc() + n] = acc;
        }
    return out;
}

nat::GemmDescriptor desc_of(
        const fp32_call_t &c, const fp32_operands_t &o, bool is_weights_const) {
    nat::GemmDescriptor d;
    d.M = c.M;
    d.N = c.N;
    d.K = c.K;
    d.lda = o.lda;
    d.ldb = o.ldb;
    d.ldc = c.ldc();
    d.transB = c.transB;
    d.alpha = c.alpha;
    d.beta = 0.0f;
    d.src_dt = d.wei_dt = d.dst_dt = kF32;
    d.bias_dt = c.bias ? kF32 : data_type_t::none;
    d.bias = c.bias ? o.bias.data() : nullptr;
    d.is_weights_const = is_weights_const;
    d.num_threads = c.threads;
    return d;
}

mm::matmul_params params_of(const fp32_call_t &c) {
    mm::matmul_params p;
    p.dtypes.src = p.dtypes.wei = p.dtypes.dst = kF32;
    p.dtypes.bias = c.bias ? kF32 : data_type_t::none;
    p.num_threads = c.threads;
    if (c.relu) {
        mm::matmul_post_op po;
        po.po_type = post_op_type_t::relu;
        po.alpha = 0.0f;
        p.postop_.push_back(po);
    }
    return p;
}

/// The call on the native BRGEMM, the weight packed for it alone.
std::vector<float> run_native(const fp32_call_t &c, const fp32_operands_t &o) {
    std::vector<float> out(static_cast<size_t>(c.M) * c.ldc(), kUnwritten);
    nat::native_weight_ref_t ref;
    ref.weight = o.wei.data();
    ref.pack_nr = kNr;
    ref.weight_cache_mode = 0;
    const mm::matmul_params p = params_of(c);
    EXPECT_EQ(
            nat::native_brgemm_execute(desc_of(c, o, false), ref, o.src.data(),
                    out.data(), c.bias ? o.bias.data() : nullptr, p),
            status_t::success)
            << str(c);
    return out;
}

std::vector<fp32_call_t> bit_calls() {
    std::vector<fp32_call_t> v;
    const int shapes[][3] = {{1, 32, 64}, {5, 96, 1}, {6, 64, 7}, {7, 160, 300},
            {13, 32, 1031}, {100, 256, 512}, {300, 96, 70}, {40, 64, 9000}};
    for (const auto &s : shapes)
        for (const bool transB : {false, true})
            for (const float alpha : {1.0f, 1.5f})
                for (const int variant : {0, 1, 2})
                    for (const int threads : {1, 4}) {
                        fp32_call_t c;
                        c.M = s[0];
                        c.N = s[1];
                        c.K = s[2];
                        c.transB = transB;
                        c.alpha = alpha;
                        c.bias = variant >= 1;
                        c.relu = variant == 2;
                        c.threads = threads;
                        v.push_back(c);
                        // The store-only call with dst rows padded.
                        if (variant == 0 && alpha == 1.0f) {
                            c.ldc_pad = 7;
                            v.push_back(c);
                        }
                    }
    return v;
}

/// Equal bits, or the first element that differs.
::testing::AssertionResult same_bits(
        const std::vector<float> &got, const std::vector<float> &want) {
    for (size_t i = 0; i < want.size(); ++i)
        if (bits_of(got[i]) != bits_of(want[i]))
            return ::testing::AssertionFailure()
                    << "element " << i << ": " << got[i] << " for " << want[i];
    return ::testing::AssertionSuccess();
}

// The native BRGEMM adds each output's products in K order from zero,
// whatever its tiles, K passes and threads, also when a call left with only
// the store accumulates in the destination: the bits of the products added
// in K order from +0.0, one fused multiply-add each, with alpha, an f32 bias
// and relu after the sum, and the padding of dst rows untouched.
TEST(NativeBrgemmFp32, OutputsAddProductsInKOrder) {
    SKIP_WITHOUT_FP32_ISA();
    uint32_t seed = 500;
    for (const fp32_call_t &c : bit_calls()) {
        SCOPED_TRACE(str(c));
        const fp32_operands_t o = operands_of(c, seed += 7);
        EXPECT_TRUE(same_bits(run_native(c, o), model_output(c, o)));
    }
}

// ── matmul_direct under ALGO 10 ──────────────────────────────────────────

/// An FP32 call of matmul_direct on a weight of its own.
struct direct_t {
    buffer_t src, wei, dst;
    nt::call_t call;
};

direct_t make_direct(
        int M, int N, int K, data_type_t dst, uint32_t seed, int32_t wc = 1) {
    direct_t d;
    d.src = nt::random_buffer(kF32, static_cast<size_t>(M) * K, seed, -1, 1);
    d.wei = nt::kept_until_exit(nt::random_buffer(
            kF32, static_cast<size_t>(K) * N, seed + 1, -1, 1));
    d.dst = nt::make_buffer(static_cast<size_t>(M) * N * nt::dt_bytes(dst));
    d.call.M = M;
    d.call.N = N;
    d.call.K = K;
    d.call.src = d.src.get();
    d.call.wei = d.wei.get();
    d.call.dst = d.dst.get();
    d.call.params.dtypes.src = d.call.params.dtypes.wei = kF32;
    d.call.params.dtypes.dst = dst;
    d.call.params.weight_cache_type = wc;
    return d;
}

// Under ALGO 10 an FP32 call runs on the native BRGEMM, within the FP32
// band of the reference algorithm, on the native layout's blocks and on a
// layout padded past N.
TEST(NativeBrgemmFp32Route, Algo10ServesFp32Calls) {
    SKIP_WITHOUT_FP32_ISA();
    const WeightCacheGuard mode(2);
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    struct route_t {
        int M, N, K;
    };
    const route_t routes[] = {{1, 64, 128}, {33, 96, 1031}, {8, 4096, 512},
            {8, 48, 64}, {1, 1, 200}, {33, 100, 1031}};
    uint32_t seed = 70;
    for (const route_t &r : routes) {
        SCOPED_TRACE("M=" + std::to_string(r.M) + " N=" + std::to_string(r.N)
                + " K=" + std::to_string(r.K));
        direct_t d = make_direct(r.M, r.N, r.K, kF32, seed += 3);
        ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
        EXPECT_NE(nt::native_layouts(d.wei.get()), 0u);
        const buffer_t ref = nt::make_buffer(d.dst.bytes);
        ASSERT_EQ(nt::run_reference(d.call, d.wei.get(), ref.get()),
                status_t::success);
        EXPECT_TRUE(nt::within_band(d.dst.get(), r.N, ref.get(), r.N, kF32, r.M,
                r.N, rtol_f32,
                nt::comparison_abs_bound(
                        r.K, epsilon_f32, 1.0f, 0.0f, false, false, false)));
    }
    mm::clear_matmul_weight_caches();
}

/// v rounded to the nearest bf16, ties to even.
uint16_t bf16_rne(float v) {
    const uint32_t u = bits_of(v);
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// A bf16 dst takes each output of the f32 call rounded to nearest even, and
// the padding of its rows stays untouched.
TEST(NativeBrgemmFp32Route, Bf16DstRoundsTheF32Output) {
    SKIP_WITHOUT_FP32_ISA();
    uint32_t seed = 300;
    for (fp32_call_t c : bit_calls()) {
        if (c.threads != 4 || c.alpha != 1.0f) continue;
        SCOPED_TRACE(str(c));
        const fp32_operands_t o = operands_of(c, seed += 5);
        const std::vector<float> f32_out = run_native(c, o);
        nat::GemmDescriptor d = desc_of(c, o, false);
        d.dst_dt = kBf16;
        mm::matmul_params p = params_of(c);
        p.dtypes.dst = kBf16;
        nat::native_weight_ref_t ref;
        ref.weight = o.wei.data();
        ref.pack_nr = kNr;
        ref.weight_cache_mode = 0;
        constexpr uint16_t kUnwrittenBf16 = 0x7f7f;
        std::vector<uint16_t> out(
                static_cast<size_t>(c.M) * c.ldc(), kUnwrittenBf16);
        ASSERT_EQ(nat::native_brgemm_execute(d, ref, o.src.data(), out.data(),
                          c.bias ? o.bias.data() : nullptr, p),
                status_t::success);
        int differ = 0;
        for (size_t i = 0; i < out.size(); ++i)
            differ += static_cast<int>(i % c.ldc()) < c.N
                    ? out[i] != bf16_rne(f32_out[i])
                    : out[i] != kUnwrittenBf16;
        EXPECT_EQ(differ, 0) << "of " << out.size() << " elements";
    }
}

// The native BRGEMM turns an FP32 call with a dst other than f32 / bf16
// away before any work: unimplemented, and dst untouched, also through
// matmul_direct, with lowoha_algo native_brgemm, for the s8, u8 and s32 dsts
// it takes with an f32 source.
TEST(NativeBrgemmFp32Route, OtherDstTypesAreTurnedAway) {
    SKIP_WITHOUT_FP32_ISA();
    fp32_call_t c;
    c.M = 4;
    c.N = 64;
    c.K = 32;
    const fp32_operands_t o = operands_of(c, 41);
    for (const data_type_t dst : {data_type_t::f16, data_type_t::s8,
                 data_type_t::u8, data_type_t::s32}) {
        nat::GemmDescriptor d = desc_of(c, o, false);
        d.dst_dt = dst;
        mm::matmul_params p = params_of(c);
        p.dtypes.dst = dst;
        nat::native_weight_ref_t ref;
        ref.weight = o.wei.data();
        ref.pack_nr = kNr;
        ref.weight_cache_mode = 0;
        std::vector<float> out(static_cast<size_t>(c.M) * c.N, kUnwritten);
        EXPECT_EQ(nat::native_brgemm_execute(
                          d, ref, o.src.data(), out.data(), nullptr, p),
                status_t::unimplemented);
        EXPECT_TRUE(std::all_of(out.begin(), out.end(),
                [](float v) { return v == kUnwritten; }));
    }
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    for (const data_type_t dst :
            {data_type_t::s8, data_type_t::u8, data_type_t::s32}) {
        SCOPED_TRACE(zendnnl::common::dtype_info(dst));
        direct_t d = make_direct(c.M, c.N, c.K, dst, 43);
        d.call.params.lowoha_algo = matmul_algo_t::native_brgemm;
        EXPECT_TRUE(nt::refused_with(d.call, status_t::unimplemented));
    }
}

// The weight cache serves an FP32 weight like an s8 one: mode 1 and mode 2
// keep one layout out of place (an f32 weight is never rewritten in place),
// later calls of the thread take it from the memo, and mode 0 keeps none.
TEST(NativeBrgemmFp32Route, WeightCacheKeepsTheLayoutOutOfPlace) {
    SKIP_WITHOUT_FP32_ISA();
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    for (const int32_t wc : {0, 1, 2}) {
        SCOPED_TRACE("weight cache " + std::to_string(wc));
        WeightCacheGuard cache(wc);
        direct_t d = make_direct(4, 128, 256, kF32, 90 + wc, wc);
        const std::vector<float> raw(
                d.wei.as<float>(), d.wei.as<float>() + d.wei.bytes / 4);
        std::vector<float> first;
        for (int call = 0; call < 2; ++call) {
            ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
            const float *out = d.dst.as<float>();
            if (call == 0)
                first.assign(out, out + d.dst.bytes / 4);
            else
                EXPECT_EQ(std::memcmp(first.data(), out, d.dst.bytes), 0);
        }
        EXPECT_EQ(std::memcmp(raw.data(), d.wei.get(), d.wei.bytes), 0)
                << "the raw weight was rewritten";
        EXPECT_FALSE(
                nat::native_weight_packed_in_place(d.wei.get(), d.wei.bytes));
        EXPECT_EQ(nt::native_layouts(d.wei.get()), wc == 0 ? 0u : 1u);
        EXPECT_EQ(nat::native_weight_memo_entries(d.wei.get()),
                wc == 0 ? 0u : 1u);
    }
    mm::clear_matmul_weight_caches();
}

} // namespace
