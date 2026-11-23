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

// F16 in the native BRGEMM (ALGO 10): the kernels of every row count and
// width over K tails against a scalar model (on a CPU with AVX512-FP16 also
// the F16_PH kernels, against the model of their blocked f16 sums and their
// error bound), the f16 native layout and the raw weight rebuilt from it,
// the plans of F16 problems, the outputs bit for bit as the kernels add the
// products (the widened inputs in K order, or the blocked f16 sums of the
// F16_PH kernels) for every N tail, one row, K passes, transposed sources,
// alpha and beta, f16 / f32 / bf16 biases, relu and binary post-ops, f16
// and f32 dsts and any plan, gated activations and other post-ops within
// the band of that model, the F16 calls matmul_direct takes to the native
// BRGEMM with or without AVX512-FP16, at weight cache modes 0, 1 and 2, and
// the kernels ZENDNNL_MATMUL_NATIVE_F16_ACCUM selects.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

#include <gtest/gtest.h>

#include "common/float16.hpp"
#include "common/platform_info.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/f16/native_f16_ph_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/f16/native_f16_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_file.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_view.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using nt::buffer_t;
using zendnnl::common::float16_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;
using gated_t = mm::matmul_gated_act_t;

constexpr data_type_t kF16 = data_type_t::f16;
constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kNone = data_type_t::none;
constexpr int kNr = nat::kNativeLayoutPackNr;

#define SKIP_WITHOUT_NATIVE_ISA() \
    do { \
        if (!nat::native_base_isa()) \
            GTEST_SKIP() << "needs AVX-512 F, BW, VL and DQ and FMA"; \
    } while (0)

bool has_avx512_fp16() {
    return zendnnl::common::zendnnl_platform_info().get_avx512_f16_status();
}

/// v rounded to the nearest f16, ties to even.
uint16_t f16_of(float v) {
    return float16_t::f32_to_f16_val(v);
}

/// The f16 h widened to f32, exactly.
float f32_of(uint16_t h) {
    return float16_t::f16_to_f32_val(h);
}

/// The bf16 h widened to f32, exactly.
float bf16_value(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

uint32_t bits_of(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

/// Whether this process runs f16 calls on the F16_PH kernels: a CPU with
/// AVX512-FP16, a build that has them, and ZENDNNL_MATMUL_NATIVE_F16_ACCUM
/// other than f32 (in any case).
bool f16_ph_calls() {
    return nat::native_kernel_registry().f16_kind()
            == nat::native_kernel_kind_t::f16_ph;
}

/// The f16 nearest to v + e, ties to even, as a float (infinite past the f16
/// range): v is that exact sum rounded to double, e what the rounding left.
float round_to_f16(double v, double e) {
    if (v == 0.0) return static_cast<float>(v);
    const double x = std::fabs(v);
    const double up = v > 0.0 ? e : -e;
    int exp2 = 0;
    std::frexp(x, &exp2);
    // The quantum of x's binade, 2^-24 below the normal range.
    const double q = std::ldexp(1.0, std::max(exp2 - 11, -24));
    double n = std::floor(x / q);
    const double frac = x / q - n;
    if (frac > 0.5
            || (frac == 0.5
                    && (up > 0.0
                            || (up == 0.0
                                    && static_cast<int64_t>(n) % 2 != 0))))
        n += 1.0;
    const double r
            = n * q > 65504.0 ? std::numeric_limits<double>::infinity() : n * q;
    return static_cast<float>(v > 0.0 ? r : -r);
}

/// a * b + c of f16 values rounded once to f16, ties to even: one f16 fused
/// multiply-add of AVX512-FP16.  The product is exact in double and the sum's
/// rounding error, exact too (TwoSum), breaks a tie.
float f16_fma(float a, float b, float c) {
    const double p = static_cast<double>(a) * b;
    const double s = p + c;
    const double t = s - p;
    const double e = (p - (s - t)) + (static_cast<double>(c) - t);
    return round_to_f16(s, e);
}

/// The sum over k below K of a(k) * b(k) (f16 values) from +0.0 as the
/// kernels of f16 calls take it: blocked, as the F16_PH kernels, the f32 sum
/// in order of the blocks of kNativeF16AccumBlockK steps from k 0, each the
/// f16 fused multiply-adds of its products in K order from +0.0; else, as
/// the F16 kernels, one f32 fused multiply-add per k in K order.
template <typename A, typename B>
float f16_dot(int K, const A &a, const B &b, bool blocked, float sum = 0.0f) {
    if (!blocked) {
        for (int k = 0; k < K; ++k)
            sum = std::fma(a(k), b(k), sum);
        return sum;
    }
    for (int k0 = 0; k0 < K; k0 += nat::kNativeF16AccumBlockK) {
        const int k1 = std::min(K, k0 + nat::kNativeF16AccumBlockK);
        float acc = 0.0f;
        for (int k = k0; k < k1; ++k)
            acc = f16_fma(a(k), b(k), acc);
        sum = sum + acc;
    }
    return sum;
}

std::vector<uint16_t> random_f16(
        size_t n, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    const buffer_t b = nt::random_buffer(kF16, n, seed, lo, hi);
    return std::vector<uint16_t>(b.as<uint16_t>(), b.as<uint16_t>() + n);
}

std::vector<float> random_floats(
        size_t n, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    const buffer_t b = nt::random_buffer(kF32, n, seed, lo, hi);
    return std::vector<float>(b.as<float>(), b.as<float>() + n);
}

// ── Kernels ──────────────────────────────────────────────────────────────

/// The tile the kernel computes: beta * C rounded once (+0.0 when beta is
/// 0), then one fused multiply-add per k in K order of A and B widened.
void model_tile(const std::vector<float> &A, int lda,
        const std::vector<uint16_t> &B, int b_stride, std::vector<float> &C,
        int ldc, int mr, int nr, int K, float beta) {
    for (int m = 0; m < mr; ++m)
        for (int n = 0; n < nr; ++n) {
            float acc = beta != 0.0f ? beta * C[m * ldc + n] : 0.0f;
            for (int k = 0; k < K; ++k)
                acc = std::fma(A[static_cast<size_t>(m) * lda + k],
                        f32_of(B[static_cast<size_t>(k) * b_stride + n]), acc);
            C[m * ldc + n] = acc;
        }
}

// Every row count 1 .. 12 at both widths, over K of every residue of the
// four-step loop and K blocks that leave a tail, with beta 0 (C holds NaN,
// which must not be read), 1 and others: the model's bits, and nothing
// outside the tile written.
TEST(NativeBrgemmF16Kernel, EveryRowCountWidthAndKTailMatchesTheModel) {
    SKIP_WITHOUT_NATIVE_ISA();
    const int Ks[] = {1, 2, 3, 4, 5, 7, 8, 13, 64, 67, 259};
    const float betas[] = {0.0f, 1.0f, 0.5f, -1.25f};
    int cases = 0, failures = 0;
    for (const int nv : {1, 2}) {
        const int nr = 16 * nv;
        for (const int b_stride : {kNr, nr + 24})
            for (int mr = 1; mr <= 12; ++mr) {
                const nat::native_f16_ukernel_fn_t fn
                        = nat::select_native_f16_ukernel(mr, nv);
                ASSERT_NE(fn, nullptr) << "mr " << mr << " nv " << nv;
                for (const int K : Ks)
                    for (const int BK : {K, 4, 5, 16})
                        for (const float beta : betas) {
                            const int lda = K + 3, ldc = nr + 5;
                            const uint32_t seed = static_cast<uint32_t>(
                                    mr * 1000 + K * 10 + nv);
                            const std::vector<float> A = random_floats(
                                    static_cast<size_t>(mr) * lda, seed);
                            const std::vector<uint16_t> B = random_f16(
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
        EXPECT_EQ(nat::select_native_f16_ukernel(1, nv), nullptr);
    for (const int mr : {0, 13})
        EXPECT_EQ(nat::select_native_f16_ukernel(mr, 1), nullptr);
}

// The registry offers the F16 kernels at 16 and 32 columns, every row count
// 1 .. 12, on a CPU with AVX-512 F, and tiles of 64 and 128 columns of
// 32-column slabs, which take the kernels of 32.
TEST(NativeBrgemmF16Kernel, RegistryServesEveryRowCount) {
    const nat::native_kernel_registry_t &r = nat::native_kernel_registry();
    for (const int nr : {16, 32, 64, 128}) {
        const nat::native_kernel_caps_t *c
                = r.caps(nat::native_kernel_kind_t::f16, nr);
        if (!nat::detect_uarch().avx512f) {
            EXPECT_EQ(c, nullptr);
            EXPECT_EQ(r.f16_table(nr), nullptr);
            continue;
        }
        ASSERT_NE(c, nullptr) << nr;
        EXPECT_EQ(c->max_mr, 12);
        for (int mr = 1; mr <= 12; ++mr)
            EXPECT_EQ(r.f16_table(nr)[mr],
                    nat::select_native_f16_ukernel(mr, std::min(nr, 32) / 16))
                    << nr << " " << mr;
        EXPECT_TRUE(r.covers(nat::native_kernel_kind_t::f16, 12, nr));
        EXPECT_FALSE(r.covers(nat::native_kernel_kind_t::f16, 13, nr));
    }
    for (const int nr : {48, 96, 256}) {
        EXPECT_EQ(r.caps(nat::native_kernel_kind_t::f16, nr), nullptr) << nr;
        EXPECT_EQ(r.f16_table(nr), nullptr) << nr;
    }
}

#define SKIP_WITHOUT_F16_PH_KERNELS() \
    do { \
        if (!nat::native_base_isa() || !nat::detect_uarch().avx512fp16) \
            GTEST_SKIP() << "needs AVX512-FP16"; \
        if (nat::select_native_f16_ph_ukernel(1, 2) == nullptr) \
            GTEST_SKIP() << "a build without the AVX512-FP16 kernels"; \
    } while (0)

/// The element (k, n) of an F16_PH kernel's B: rows b_stride apart, columns
/// 32 .. 63 b_next elements after columns 0 .. 31.
size_t ph_b_at(int k, int n, int b_stride, size_t b_next) {
    return (n < kNr ? 0 : b_next) + static_cast<size_t>(k) * b_stride + n % kNr;
}

/// The tile the F16_PH kernel computes from f16 A rows: per element the f32
/// sum from beta * C rounded once (+0.0 when beta is 0) of the blocked f16
/// sums of f16_dot.
void model_ph_tile(const std::vector<uint16_t> &A, int lda,
        const std::vector<uint16_t> &B, int b_stride, size_t b_next,
        std::vector<float> &C, int ldc, int mr, int nr, int K, float beta) {
    for (int m = 0; m < mr; ++m)
        for (int n = 0; n < nr; ++n) {
            const float s = beta != 0.0f ? beta * C[m * ldc + n] : 0.0f;
            C[m * ldc + n] = f16_dot(K, [&](int k) {
                return f32_of(A[static_cast<size_t>(m) * lda + k]);
            }, [&](int k) {
                return f32_of(B[ph_b_at(k, n, b_stride, b_next)]);
            }, true, s);
        }
}

/// n * u / (1 - n * u): the bound of the relative error of a recursive sum
/// of n terms rounded at unit roundoff u, against the sum of their
/// magnitudes.
double gamma_of(int64_t n, double u) {
    return static_cast<double>(n) * u / (1.0 - static_cast<double>(n) * u);
}

/// What an F16_PH sum of K products may be off by from the exact sum: within
/// each block of kNativeF16AccumBlockK steps the f16 multiply-adds round at
/// 2^-11, so a block is off by at most gamma(block, 2^-11) times the sum of
/// its products' magnitudes; the f32 sums of the blocks and of beta * C
/// round at 2^-24, gamma(blocks + 1, 2^-24) of their magnitudes.  mag: the
/// sum of the products' magnitudes; base: |beta * C|.
double f16_ph_bound(int K, double mag, double base) {
    const int64_t blocks
            = (K + nat::kNativeF16AccumBlockK - 1) / nat::kNativeF16AccumBlockK;
    const double block = gamma_of(nat::kNativeF16AccumBlockK, 0x1p-11);
    const double sums = gamma_of(blocks + 2, 0x1p-24);
    return block * mag + sums * ((1.0 + block) * mag + base) + 0x1p-24 * base;
}

// The F16_PH kernels of every row count 1 .. 12 at the three widths (16,
// 32, and 64 over two blocks), over K of one, around the block
// (kNativeF16AccumBlockK - 1, kNativeF16AccumBlockK, + 1), several blocks
// and a tail (3 kNativeF16AccumBlockK + 5, 4103) and 32768 for some rows,
// with beta 0 (C holds NaN, which must not be read) and 1 (a later K pass):
// the bits of the blocked model, nothing outside the tile written, and
// every element within f16_ph_bound of the f64 sum.
TEST(NativeBrgemmF16PhKernel, EveryRowCountWidthAndKMatchesTheBlockedModel) {
    SKIP_WITHOUT_F16_PH_KERNELS();
    constexpr int kB = nat::kNativeF16AccumBlockK;
    const int Ks[] = {1, kB - 1, kB, kB + 1, 3 * kB + 5, 4103, 32768};
    int cases = 0, failures = 0, outside = 0;
    for (const int nv : {1, 2, 4}) {
        const int nr = 16 * nv;
        for (int mr = 1; mr <= 12; ++mr) {
            const nat::native_f16_ph_ukernel_fn_t fn
                    = nat::select_native_f16_ph_ukernel(mr, nv);
            ASSERT_NE(fn, nullptr) << "mr " << mr << " nv " << nv;
            for (const int K : Ks) {
                if (K == 32768 && mr != 1 && mr != 7 && mr != 12) continue;
                for (const float beta : {0.0f, 1.0f}) {
                    const int lda = K + 3, ldc = nr + 5;
                    const int b_stride
                            = mr % 2 == 0 ? kNr : std::min(nr, kNr) + 24;
                    // The second block of a tile of 64 columns, off any
                    // alignment.
                    const size_t b_next = nv == 4
                            ? static_cast<size_t>(K) * b_stride + 7
                            : 0;
                    const uint32_t seed
                            = static_cast<uint32_t>(mr * 1000 + K * 10 + nv);
                    const std::vector<uint16_t> A
                            = random_f16(static_cast<size_t>(mr) * lda, seed);
                    const std::vector<uint16_t> B = random_f16(
                            b_next + static_cast<size_t>(K) * b_stride,
                            seed + 1);
                    std::vector<float> C0 = random_floats(
                            static_cast<size_t>(mr) * ldc, seed + 2);
                    if (beta == 0.0f)
                        for (int m = 0; m < mr; ++m)
                            for (int n = 0; n < nr; ++n)
                                C0[m * ldc + n] = std::numeric_limits<
                                        float>::quiet_NaN();
                    std::vector<float> got = C0, want = C0;
                    fn(A.data(), lda, B.data(), b_stride, b_next, got.data(),
                            ldc, K, beta);
                    model_ph_tile(A, lda, B, b_stride, b_next, want, ldc, mr,
                            nr, K, beta);
                    ++cases;
                    bool ok = true;
                    for (size_t i = 0; i < got.size() && ok; ++i)
                        ok = bits_of(got[i]) == bits_of(want[i]);
                    if (!ok && failures++ < 10)
                        ADD_FAILURE() << "mr " << mr << " nr " << nr << " K "
                                      << K << " beta " << beta;
                    for (int m = 0; m < mr; ++m)
                        for (int n = 0; n < nr; ++n) {
                            const uint16_t *a
                                    = &A[static_cast<size_t>(m) * lda];
                            double exact = 0.0, mag = 0.0;
                            for (int k = 0; k < K; ++k) {
                                const double p
                                        = static_cast<double>(f32_of(a[k]))
                                        * f32_of(B[ph_b_at(
                                                k, n, b_stride, b_next)]);
                                exact += p;
                                mag += std::fabs(p);
                            }
                            const double base = beta != 0.0f
                                    ? static_cast<double>(beta)
                                            * C0[m * ldc + n]
                                    : 0.0;
                            const double err = std::fabs(
                                    got[m * ldc + n] - (base + exact));
                            outside += !(err
                                    <= f16_ph_bound(K, mag, std::fabs(base)));
                        }
                }
            }
        }
    }
    EXPECT_EQ(failures, 0) << "of " << cases << " cases";
    EXPECT_EQ(outside, 0) << "elements past the bound";
    for (const int nv : {0, 3, 8})
        EXPECT_EQ(nat::select_native_f16_ph_ukernel(1, nv), nullptr);
    for (const int mr : {0, 13})
        EXPECT_EQ(nat::select_native_f16_ph_ukernel(mr, 4), nullptr);
}

// Long sums need the blocks: over K 32768 a sum of ones in f16 stops at
// 2048, where adding one rounds back to it, and a sum of positive products
// stops near the magnitude where they fall below half its last bit; in
// blocks of kNativeF16AccumBlockK steps the kernel gives the 32768 ones
// exactly and the positive products within f16_ph_bound of the f64 sum,
// while the f16 sum over all of K is far outside it.
TEST(NativeBrgemmF16PhKernel, BlocksBoundTheErrorOfLongSums) {
    SKIP_WITHOUT_F16_PH_KERNELS();
    const int K = 32768, nr = 32, mr = 2;
    const nat::native_f16_ph_ukernel_fn_t fn
            = nat::select_native_f16_ph_ukernel(mr, 2);
    ASSERT_NE(fn, nullptr);
    std::vector<uint16_t> A(static_cast<size_t>(mr) * K, f16_of(1.0f));
    std::vector<uint16_t> B(static_cast<size_t>(K) * kNr, f16_of(1.0f));
    std::vector<float> C(static_cast<size_t>(mr) * nr, -1.0f);
    fn(A.data(), K, B.data(), kNr, 0, C.data(), nr, K, 0.0f);
    float whole = 0.0f;
    for (int k = 0; k < K; ++k)
        whole = f16_fma(1.0f, 1.0f, whole);
    EXPECT_EQ(whole, 2048.0f);
    for (const float v : C)
        EXPECT_EQ(v, static_cast<float>(K));

    const std::vector<uint16_t> pa
            = random_f16(static_cast<size_t>(mr) * K, 7700, 0.0f, 1.0f);
    const std::vector<uint16_t> pb
            = random_f16(static_cast<size_t>(K) * kNr, 7701, 0.0f, 1.0f);
    fn(pa.data(), K, pb.data(), kNr, 0, C.data(), nr, K, 0.0f);
    int outside = 0, whole_inside = 0;
    for (int m = 0; m < mr; ++m)
        for (int n = 0; n < nr; ++n) {
            double exact = 0.0;
            float f16_sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float a = f32_of(pa[static_cast<size_t>(m) * K + k]);
                const float b = f32_of(pb[static_cast<size_t>(k) * kNr + n]);
                exact += static_cast<double>(a) * b;
                f16_sum = f16_fma(a, b, f16_sum);
            }
            const double bound = f16_ph_bound(K, exact, 0.0);
            outside += !(std::fabs(C[m * nr + n] - exact) <= bound);
            whole_inside += std::fabs(f16_sum - exact) <= bound;
        }
    EXPECT_EQ(outside, 0);
    EXPECT_EQ(whole_inside, 0);
}

// The registry offers the F16_PH kernels at 16, 32 and 64 columns, every
// row count 1 .. 12, and tiles of 128 columns of two calls of 64, exactly on
// a CPU with AVX512-FP16 in a build that has them unless
// ZENDNNL_MATMUL_NATIVE_F16_ACCUM is f32 (in any case), and f16 calls then
// take them; otherwise it has none and f16 calls take the F16 kernels.
TEST(NativeBrgemmF16PhKernel, RegistryServesThemWhereTheCpuHasAvx512Fp16) {
    const nat::native_kernel_registry_t &r = nat::native_kernel_registry();
    const bool want = nt::f16_ph_kernels_registered();
    for (const int nr : {16, 32, 64, 128}) {
        const nat::native_kernel_caps_t *c
                = r.caps(nat::native_kernel_kind_t::f16_ph, nr);
        if (!want) {
            EXPECT_EQ(c, nullptr) << nr;
            EXPECT_EQ(r.f16_ph_table(nr), nullptr) << nr;
            continue;
        }
        ASSERT_NE(c, nullptr) << nr;
        EXPECT_EQ(c->max_mr, 12);
        for (int mr = 1; mr <= 12; ++mr)
            EXPECT_EQ(r.f16_ph_table(nr)[mr],
                    nat::select_native_f16_ph_ukernel(
                            mr, std::min(nr, 64) / 16))
                    << nr << " " << mr;
    }
    for (const int nr : {48, 96, 256})
        EXPECT_EQ(r.caps(nat::native_kernel_kind_t::f16_ph, nr), nullptr) << nr;
    EXPECT_EQ(r.f16_kind(),
            want ? nat::native_kernel_kind_t::f16_ph
                 : nat::native_kernel_kind_t::f16);
    EXPECT_EQ(f16_ph_calls(), want);
}

// ── Layout ───────────────────────────────────────────────────────────────

// Each element of an f16 weight at its place in the layout: block p / 32,
// row k, column p % 32 for pack column p, from [K, N] or [N, K] storage,
// with a padded ldb, odd K (the [N, K] pairs of the pack), K past the tiled
// walk of a [K, N] weight, a last block of fewer columns (zero past N) and
// more than one thread.
TEST(NativeBrgemmF16Layout, PackPlacesEveryElement) {
    SKIP_WITHOUT_NATIVE_ISA();
    int checked = 0;
    for (const int K : {1, 2, 3, 64, 1024, 1031})
        for (const int N : {32, 48, 33, 96, 544})
            for (const bool transB : {false, true})
                for (const int pad : {0, 5})
                    for (const int threads : {1, 4}) {
                        if (K >= 1024 && N == 544 && threads == 1) continue;
                        SCOPED_TRACE("K=" + std::to_string(K)
                                + " N=" + std::to_string(N)
                                + " transB=" + std::to_string(transB)
                                + " pad=" + std::to_string(pad)
                                + " threads=" + std::to_string(threads));
                        const int ldb = (transB ? K : N) + pad;
                        std::vector<uint16_t> w(
                                static_cast<size_t>(transB ? N : K) * ldb);
                        for (size_t i = 0; i < w.size(); ++i)
                            w[i] = static_cast<uint16_t>(1 + i % 65521);
                        const int padded = nat::native_padded_n(N, kNr);
                        const size_t bytes
                                = nat::native_packed_bytes(kF16, K, N, kNr);
                        ASSERT_EQ(bytes,
                                (static_cast<size_t>(K) * padded * 2 + 63)
                                        & ~size_t {63});
                        ASSERT_EQ(nat::native_block_bytes(kF16, K, kNr),
                                static_cast<size_t>(K) * kNr * 2);
                        std::vector<uint16_t> packed(bytes / 2, 0xFFFF);
                        ASSERT_EQ(nat::native_pack_weight(kF16, w.data(), K, N,
                                          ldb, kNr, transB, packed.data(),
                                          threads),
                                status_t::success);
                        int bad = 0;
                        for (int p = 0; p < padded; ++p) {
                            for (int k = 0; k < K; ++k) {
                                const size_t at
                                        = static_cast<size_t>(p / kNr) * K * kNr
                                        + static_cast<size_t>(k) * kNr
                                        + p % kNr;
                                if (p >= N) {
                                    bad += packed[at] != 0;
                                    continue;
                                }
                                const size_t from = transB
                                        ? static_cast<size_t>(p) * ldb + k
                                        : static_cast<size_t>(k) * ldb + p;
                                bad += packed[at] != w[from];
                            }
                        }
                        EXPECT_EQ(bad, 0);
                        ++checked;
                    }
    EXPECT_GT(checked, 0);
}

// The unpack of an f16 image writes back every weight element the pack
// placed, from [K, N] or [N, K] storage, and leaves the padding of a padded
// ldb as it was.
TEST(NativeBrgemmF16Layout, UnpackRebuildsTheWeight) {
    SKIP_WITHOUT_NATIVE_ISA();
    for (const int K : {1, 3, 64, 1031})
        for (const int N : {32, 96})
            for (const bool transB : {false, true})
                for (const int pad : {0, 3}) {
                    SCOPED_TRACE("K=" + std::to_string(K)
                            + " N=" + std::to_string(N)
                            + " transB=" + std::to_string(transB)
                            + " pad=" + std::to_string(pad));
                    const int ldb = (transB ? K : N) + pad;
                    const size_t elems
                            = static_cast<size_t>(transB ? N : K) * ldb;
                    const std::vector<uint16_t> w
                            = random_f16(elems, 900 + K + N);
                    std::vector<uint16_t> packed(
                            nat::native_packed_bytes(kF16, K, N, kNr) / 2);
                    ASSERT_EQ(nat::native_pack_weight(kF16, w.data(), K, N, ldb,
                                      kNr, transB, packed.data(), 4),
                            status_t::success);
                    std::vector<uint16_t> back(elems, 0x5A5A);
                    nat::native_unpack_weight(kF16, packed.data(), K, N, ldb,
                            kNr, transB, back.data(), 4);
                    int bad = 0;
                    for (size_t i = 0; i < elems; ++i) {
                        const bool element
                                = static_cast<int>(i % ldb) < (transB ? K : N);
                        bad += back[i] != (element ? w[i] : 0x5A5A);
                    }
                    EXPECT_EQ(bad, 0);
                }
}

// An f16 weight of K up to 2147483632, the largest multiple of 16 an int
// holds: the widened source rows' steps of 16 elements and the [N, K]
// pack's K pairs stay within an int.  The layout rule refuses a larger K,
// and the native BRGEMM turns such a call away before any work (no buffer
// is read).
TEST(NativeBrgemmF16Layout, KWhoseStepsExceedAnInt) {
    EXPECT_EQ(nat::kMaxNativeF16K, 2147483632);
    if (!nat::native_base_isa())
        GTEST_SKIP() << "needs AVX-512 F, BW, VL and DQ and FMA";
    const char *const reason
            = "an f16 weight of K whose steps of 16 exceed int";
    const auto rule = [](int K, bool transB) {
        nat::native_layout_query_t q;
        q.wei_dt = kF16;
        q.src_dt = kF16;
        q.K = K;
        q.N = 64;
        q.ldb = transB ? K : 64;
        q.transB = transB;
        return nat::decide_native_layout(q);
    };
    for (const bool transB : {false, true}) {
        EXPECT_TRUE(rule(nat::kMaxNativeF16K, transB).native);
        for (const int K : {nat::kMaxNativeF16K + 1,
                     std::numeric_limits<int32_t>::max()}) {
            const nat::native_layout_decision_t d = rule(K, transB);
            EXPECT_FALSE(d.native) << K;
            EXPECT_STREQ(d.reason != nullptr ? d.reason : "", reason) << K;
        }
    }
    mm::clear_matmul_weight_caches();
    const int K = std::numeric_limits<int32_t>::max(), N = 64;
    alignas(64) uint16_t buf[64] = {};
    for (const int M : {1, 2}) {
        SCOPED_TRACE("M " + std::to_string(M));
        mm::matmul_params p;
        p.dtypes.src = kF16;
        p.dtypes.wei = kF16;
        p.dtypes.dst = kF16;
        p.lowoha_algo = matmul_algo_t::native_brgemm;
        status_t status = status_t::success;
        const char *why = nullptr;
        EXPECT_TRUE(nat::native_layout_check('r', false, true, M, N, K, 1.0f,
                buf, K, buf, K, nullptr, 0.0f, buf, N, true, 1, 1,
                gated_t::none, p, status, why));
        EXPECT_EQ(status, status_t::unimplemented);
        EXPECT_STREQ(why != nullptr ? why : "", reason);
    }
}

// The layout's facts of an f16 weight: K rows per block, a group stride of
// one block row, the raw weight's bytes for N a multiple of the blocks, and
// the layout rule's shape (any N, held in whole blocks).
TEST(NativeBrgemmF16Layout, ViewAndRuleFacts) {
    EXPECT_EQ(nat::native_k_groups(kF16, 1031), 1031);
    EXPECT_TRUE(nat::native_layout_fits(kF16, 1, 32, kNr));
    EXPECT_TRUE(nat::native_layout_fits(kF16, 7, 33, kNr));
    EXPECT_FALSE(nat::native_layout_fits(kF16, 0, 64, kNr));
    EXPECT_FALSE(nat::native_layout_fits(kF16, 64, 0, kNr));
    EXPECT_EQ(nat::native_packed_bytes(kF16, 64, 48, kNr),
            static_cast<size_t>(64) * 64 * 2);
    EXPECT_EQ(nat::native_packed_bytes(kF16, 1031, 4096, kNr),
            nat::native_raw_weight_bytes(kF16, 1031, 4096, 4096, false));
    nat::native_weight_view_t v;
    v.wei_dt = kF16;
    v.pack_nr = kNr;
    EXPECT_EQ(v.group_stride(), kNr);
}

// ── Plans ────────────────────────────────────────────────────────────────

nat::native_problem_t problem(int M, int N, int K, int threads, data_type_t wei,
        data_type_t dst = kF16) {
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

// The names of f16 in ZENDNNL_MATMUL_NATIVE_CONFIG and the tuning file: a
// clause or a table entry for dtype or src f16 is taken and holds for an
// f16 problem only, the bf16 entries of the table index next to it, and
// the rule features weigh an f16 source and weight at two bytes an element.
TEST(NativeBrgemmF16Plan, TuningNamesTakeF16) {
    nat::native_problem_t f16;
    f16.M = 24;
    f16.N = 4096;
    f16.K = 2048;
    f16.pack_nr = kNr;
    f16.threads = 16;
    f16.src_dt = kF16;
    f16.wei_dt = kF16;
    f16.dst_dt = kF16;
    nat::native_problem_t bf16 = f16;
    bf16.src_dt = kBf16;
    bf16.wei_dt = kBf16;
    bf16.dst_dt = kBf16;
    const auto clause = nat::make_native_override_source(
            "dtype=f16,src=f16:nr=16,kc=3;dtype=bf16:mb=24");
    ASSERT_NE(clause, nullptr);
    const nat::native_config_entry_t *e = clause->lookup(f16);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(nat::format_native_config(e->config), "nr=16,kc=3");
    e = clause->lookup(bf16);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(nat::format_native_config(e->config), "mb=24");

    nat::native_hw_t hw;
    hw.uarch.l1d_bytes = 48 << 10;
    hw.uarch.l2_bytes = 1 << 20;
    hw.uarch.l3_domain_bytes = 32 << 20;
    hw.uarch.avx512f = true;
    hw.logical_cpus = 32;
    hw.smt = 1;
    hw.physical_cores = 32;
    const std::string text = R"({"format": "zendnnl-native-brgemm-tuning",
        "version": 2, "entries": [
        {"id": "f16", "match": {"dtype": "f16", "n": 4096, "k": 2048},
         "config": {"nr": 16}},
        {"id": "bf16", "match": {"dtype": "bf16", "n": 4096, "k": 2048},
         "config": {"mb": 24}}],
        "rules": {"kind": "ordered", "list": [
        {"when": [["a_bytes", "==", 98304], ["b_bytes", "==", 16777216],
                  ["esz", "==", 2]], "config": {"kc": 64}}]}})";
    const nt::temp_file_t file("f16_tuning.json", text);
    const nat::native_tuning_sources_t t
            = nat::load_native_tuning_file(file.path(), hw);
    ASSERT_NE(t.table, nullptr);
    ASSERT_NE(t.rules, nullptr);
    e = t.table->lookup(f16);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(nat::format_native_config(e->config), "nr=16");
    e = t.table->lookup(bf16);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(nat::format_native_config(e->config), "mb=24");
    e = t.rules->lookup(f16);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(nat::format_native_config(e->config), "kc=64");
}

// An F16 plan takes tiles of 16 or 32 columns, any K pass of at least one
// and the F16 kernels' rows, with the K pass of a two-byte weight, and tiles
// of 64 and 128 columns: 32-column slabs of the F16 kernels, or for the
// F16_PH kernels their tile of two blocks and two of those.
TEST(NativeBrgemmF16Plan, ValidateTheF16Tile) {
    SKIP_WITHOUT_NATIVE_ISA();
    const nat::UarchParams &u = nat::native_plan_uarch();
    const nat::native_problem_t f = problem(64, 256, 1000, 4, kF16);
    for (const char *text : {"nr=16", "nr=32", "kc=3", "kc=999", "mr=12",
                 "nr=16,mr=12", "schedule=blocks,unit_tiles=3,kc=64", "nr=64",
                 "nr=128", "nr=128,mr=5,kc=256"}) {
        const nat::native_plan_t q
                = nat::heuristic_plan(f, u, heuristic_of(text));
        EXPECT_EQ(nat::validate_native_plan(q, f), nullptr) << text;
    }
    nat::native_plan_t q = nat::heuristic_plan(f, u, heuristic_of("nr=128"));
    nat::native_plan_t other = q;
    other.kind = f16_ph_calls() ? nat::native_kernel_kind_t::f16
                                : nat::native_kernel_kind_t::f16_ph;
    EXPECT_EQ(nat::validate_native_plan(other, f) == nullptr, f16_ph_calls());
    q = nat::heuristic_plan(
            problem(64, 288, 1000, 4, kF16), u, heuristic_of("nr=128"));
    EXPECT_NE(nat::validate_native_plan(q, problem(64, 288, 1000, 4, kF16)),
            nullptr);
    q = nat::heuristic_plan(f, u, {});
    q.kc = 0;
    q.scratch_floats = nat::native_plan_scratch_floats(q, f);
    EXPECT_NE(nat::validate_native_plan(q, f), nullptr);
    const nat::heuristic_params_t hp = heuristic_of(
            "makespan=0,l1_pass=0,traffic_groups=0,bf16_alias=0,"
            "bf16_narrow_n=0,blocks=0,nr=32,mr=12,mb=96,threads_cap=1");
    for (const int K : {512, 4096, 9000, 65536}) {
        const nat::native_problem_t p = problem(96, 256, K, 1, kF16);
        const nat::native_plan_t r = nat::heuristic_plan(p, u, hp);
        ASSERT_EQ(nat::validate_native_plan(r, p), nullptr) << K;
        const int slab = static_cast<int>(static_cast<int64_t>(u.l2_bytes)
                * hp.bf16_l2_pct / 100 / (32 * 2));
        int want = slab < K ? std::max(hp.min_pass_k, slab & ~1) : K;
        // The F16_PH kernels take whole blocks.
        constexpr int kB = nat::kNativeF16AccumBlockK;
        if (r.kind == nat::native_kernel_kind_t::f16_ph && want < K)
            want = std::max(kB, want / kB * kB);
        EXPECT_EQ(r.kc, want) << "K " << K;
    }
}

// An F16_PH plan's K passes are whole blocks of kNativeF16AccumBlockK, so
// every block of the kernels covers the same K whatever the plan: a pinned
// pass below K is rounded down to whole blocks, at least one, as are the
// rules' passes of long K, and a plan of another pass below K is illegal;
// the F16 kernels take any pass, and the plans of the other weight types
// keep their kernels.  Without the F16_PH kernels f16 plans take the F16
// ones and a plan that names F16_PH is illegal.
TEST(NativeBrgemmF16Plan, F16PhPassesAreWholeBlocks) {
    SKIP_WITHOUT_NATIVE_ISA();
    constexpr int kB = nat::kNativeF16AccumBlockK;
    const nat::UarchParams &u = nat::native_plan_uarch();
    const int K = 9000;
    const nat::native_problem_t f = problem(64, 256, K, 4, kF16);
    // Tiles of 32 columns, which both kernel kinds take.
    nat::native_plan_t q
            = nat::heuristic_plan(f, u, heuristic_of("kc=1000,nr=32"));
    if (!f16_ph_calls()) {
        EXPECT_EQ(q.kind, nat::native_kernel_kind_t::f16);
        EXPECT_EQ(q.kc, 1000);
        EXPECT_EQ(nat::validate_native_plan(q, f), nullptr);
        q.kind = nat::native_kernel_kind_t::f16_ph;
        EXPECT_STREQ(nat::validate_native_plan(q, f),
                "no F16_PH kernel for every row count 1 .. mr at nr");
        return;
    }
    for (const int pin :
            {2, 3, kB - 1, kB, kB + 5, 1000, 4 * kB, K - 1, K, 2 * K}) {
        const std::string text = "kc=" + std::to_string(pin);
        const nat::native_plan_t p
                = nat::heuristic_plan(f, u, heuristic_of(text.c_str()));
        EXPECT_EQ(p.kind, nat::native_kernel_kind_t::f16_ph) << text;
        EXPECT_EQ(p.kc, pin >= K ? K : std::max(kB, pin / kB * kB)) << text;
        EXPECT_EQ(nat::validate_native_plan(p, f), nullptr) << text;
    }
    for (const int LK : {4096, 14336, 65536, 1000000})
        for (const char *text : {"cost_model=0", "l1_pass_max_mb=4096",
                     "kc_split=equal", "blocks=0"}) {
            const nat::native_problem_t p = problem(512, 4096, LK, 32, kF16);
            const nat::native_plan_t r
                    = nat::heuristic_plan(p, u, heuristic_of(text));
            EXPECT_EQ(nat::validate_native_plan(r, p), nullptr)
                    << LK << " " << text;
            EXPECT_TRUE(r.kc >= LK || (r.kc >= kB && r.kc % kB == 0))
                    << "K " << LK << " " << text << ": kc " << r.kc;
        }
    q.kc = kB + 1;
    q.scratch_floats = nat::native_plan_scratch_floats(q, f);
    EXPECT_STREQ(nat::validate_native_plan(q, f),
            "kc is below K and not a whole number of F16_PH blocks");
    q.kind = nat::native_kernel_kind_t::f16;
    EXPECT_EQ(nat::validate_native_plan(q, f), nullptr);
    EXPECT_EQ(
            nat::heuristic_plan(problem(64, 256, K, 4, kF32, kF32), u, {}).kind,
            nat::native_kernel_kind_t::f32);
    if (nat::detect_uarch().avx512bf16) {
        EXPECT_EQ(
                nat::heuristic_plan(problem(64, 256, K, 4, kBf16, kBf16), u, {})
                        .kind,
                nat::native_kernel_kind_t::bf16);
    }
}

// From f16_wide_min_m rows the F16_PH kernels' plans take tiles of two
// blocks (64 columns, whose kernel feeds both from one broadcast of each
// source element) where N is whole tiles of 64, narrowed to 32 and 16
// columns where tiles of 64 would be fewer than the threads, as tiles of
// 32.  Below f16_wide_min_m rows, and for a weight of N not whole tiles of
// 64, they take 32, and a plan of 64 is illegal for the latter.  The F16
// kernels take 64 only as two 32-column slabs, and the plans of other types
// do not read the key.
TEST(NativeBrgemmF16Plan, F16PhPlansTakeTilesOfTwoBlocks) {
    SKIP_WITHOUT_NATIVE_ISA();
    const nat::UarchParams &u = nat::native_plan_uarch();
    const int min_m = nat::heuristic_params_t {}.f16_wide_min_m;
    ASSERT_GT(min_m, 1);
    ASSERT_LE(min_m, 1024);
    const nat::native_problem_t wide = problem(1024, 4096, 4096, 32, kF16);
    const nat::native_problem_t below
            = problem(min_m - 1, 4096, 4096, 32, kF16);
    const nat::native_problem_t odd = problem(1024, 4128, 4096, 32, kF16);
    const nat::native_problem_t few = problem(1, 256, 4096, 32, kF16);
    const nat::native_plan_t w = nat::heuristic_plan(wide, u, {});
    const nat::native_plan_t b = nat::heuristic_plan(below, u, {});
    const nat::native_plan_t o = nat::heuristic_plan(odd, u, {});
    const nat::native_plan_t n = nat::heuristic_plan(few, u, {});
    EXPECT_EQ(nat::validate_native_plan(w, wide), nullptr);
    EXPECT_EQ(nat::validate_native_plan(b, below), nullptr);
    EXPECT_EQ(nat::validate_native_plan(o, odd), nullptr);
    EXPECT_EQ(nat::validate_native_plan(n, few), nullptr);
    EXPECT_LE(b.nr, 32);
    EXPECT_LE(o.nr, 32);
    EXPECT_LE(n.nr, 32);
    EXPECT_EQ(w.nr, f16_ph_calls() ? 64 : 32);
    const nat::native_plan_t later
            = nat::heuristic_plan(wide, u, heuristic_of("f16_wide_min_m=1025"));
    const nat::native_plan_t from_m
            = nat::heuristic_plan(wide, u, heuristic_of("f16_wide_min_m=1024"));
    EXPECT_LE(later.nr, 32);
    EXPECT_EQ(from_m.nr, w.nr);
    const nat::native_plan_t pinned
            = nat::heuristic_plan(odd, u, heuristic_of("nr=64"));
    EXPECT_NE(nat::validate_native_plan(pinned, odd), nullptr);
    nat::native_plan_t f16 = w;
    f16.kind = nat::native_kernel_kind_t::f16;
    EXPECT_EQ(nat::validate_native_plan(f16, wide), nullptr);
    const nat::native_plan_t slabs = nat::heuristic_plan(
            wide, u, heuristic_of("f16_wide_min_m=1025,nr=64"));
    EXPECT_EQ(slabs.nr, 64);
    EXPECT_EQ(nat::validate_native_plan(slabs, wide), nullptr);
    for (const nat::native_problem_t &p :
            {problem(64, 4096, 4096, 32, kBf16, kBf16),
                    problem(1024, 4096, 4096, 32, kBf16, kBf16),
                    problem(1024, 4096, 4096, 32, kF32, kF32)}) {
        const nat::native_plan_t q = nat::heuristic_plan(p, u, {});
        const nat::native_plan_t k
                = nat::heuristic_plan(p, u, heuristic_of("f16_wide_min_m=1"));
        EXPECT_EQ(std::make_tuple(k.nr, k.mr, k.mb, k.kc, k.threads),
                std::make_tuple(q.nr, q.mr, q.mb, q.kc, q.threads))
                << p.M;
    }
}

// The widened source rows the F16 kernels read are never a multiple of
// 4 KiB apart: their tiles keep tall_rows per call where the alias rule
// takes a BF16 block of rows a multiple of 4 KiB apart to bf16_alias_mr
// rows.  The F16_PH kernels read the call's source rows, which the rule
// takes as BF16 rows.
TEST(NativeBrgemmF16Plan, WidenedRowsKeepTheTallRows) {
    SKIP_WITHOUT_NATIVE_ISA();
    const nat::UarchParams &u = nat::native_plan_uarch();
    const nat::heuristic_params_t hp = heuristic_of(
            "makespan=0,l1_pass=0,traffic_groups=0,bf16_narrow_n=0,blocks=0,"
            "cost_model=0,nr=32,mb=192,threads_cap=1");
    for (const int K : {1024, 2048, 4096}) {
        const nat::native_problem_t p = problem(1024, 256, K, 1, kF16);
        const nat::native_plan_t q = nat::heuristic_plan(p, u, hp);
        ASSERT_EQ(nat::validate_native_plan(q, p), nullptr) << K;
        const bool rows_in_place
                = q.kind == nat::native_kernel_kind_t::f16_ph && K % 2048 == 0;
        EXPECT_EQ(q.mr, rows_in_place ? hp.bf16_alias_mr : hp.tall_rows)
                << "K " << K;
        if (nat::detect_uarch().avx512bf16 && K % 2048 == 0) {
            const nat::native_plan_t b = nat::heuristic_plan(
                    problem(1024, 256, K, 1, kBf16), u, hp);
            EXPECT_EQ(b.mr, hp.bf16_alias_mr) << "K " << K;
        }
    }
}

// The cost model prices an F16 plan at one FMA step a K row of two-byte
// weights: the calls of a grid it bounds take about twice the core cycles
// of the same grid of a BF16 problem, which takes a step a K pair, on the
// F16 kernels, and about the same on the F16_PH kernels, whose FMA takes a
// K row of 32 columns; and the built-in plans of F16 problems, the cost
// model's choice included, are legal and priced.
TEST(NativeBrgemmF16Plan, CostModelPricesOneFmaStepAKRow) {
    SKIP_WITHOUT_NATIVE_ISA();
    const nat::UarchParams &u = nat::native_plan_uarch();
    const nat::heuristic_params_t hp {};
    for (const int M : {1, 16, 64, 128, 1024})
        for (const int N : {4096, 28672}) {
            const nat::native_problem_t p = problem(M, N, 4096, 32, kF16);
            const nat::native_plan_t q = nat::heuristic_plan(p, u, hp);
            EXPECT_EQ(nat::validate_native_plan(q, p), nullptr)
                    << M << " " << N;
            EXPECT_GT(nat::native_plan_cost(q, p, u, hp).cycles, 0.0)
                    << M << " " << N;
        }
    const nat::heuristic_params_t pin
            = heuristic_of("nr=32,mr=12,mb=96,kc=4096,threads_cap=1");
    const nat::native_problem_t f = problem(96, 256, 4096, 1, kF16);
    nat::native_problem_t b = f;
    b.src_dt = b.wei_dt = b.dst_dt = kBf16;
    nat::native_plan_t qf = nat::heuristic_plan(f, u, pin);
    qf.kind = nat::native_kernel_kind_t::f16;
    const double cf = nat::native_plan_cost(qf, f, u, pin).core;
    const double cb
            = nat::native_plan_cost(nat::heuristic_plan(b, u, pin), b, u, pin)
                      .core;
    EXPECT_GT(cf, 1.8 * cb);
    EXPECT_LT(cf, 2.2 * cb);
    qf.kind = nat::native_kernel_kind_t::f16_ph;
    const double cp = nat::native_plan_cost(qf, f, u, pin).core;
    EXPECT_GT(cp, 0.9 * cb);
    EXPECT_LT(cp, 1.1 * cb);
}

// ── Outputs ──────────────────────────────────────────────────────────────

/// One F16 call: dst = alpha * src * wei + beta * dst (+ bias), then relu
/// and a binary post-op; dst rows ldc apart.
struct f16_call_t {
    int M = 0, N = 0, K = 0;
    bool transA = false, transB = false;
    float alpha = 1.0f, beta = 0.0f;
    data_type_t bias_dt = kNone;
    bool relu = false;
    /// 0: none; 1: binary_add of an [M, N] f16 matrix; 2: binary_mul of an
    /// f32 value per column.
    int binary = 0;
    data_type_t dst_dt = kF16;
    int threads = 1;
    int ldc_pad = 0;

    int ldc() const { return N + ldc_pad; }
    int lda() const { return transA ? M + 2 : K + 3; }
    int ldb() const { return (transB ? K : N) + 5; }
};

std::string str(const f16_call_t &c) {
    return "M=" + std::to_string(c.M) + " N=" + std::to_string(c.N) + " K="
            + std::to_string(c.K) + " transA=" + std::to_string(c.transA)
            + " transB=" + std::to_string(c.transB) + " alpha="
            + std::to_string(c.alpha) + " beta=" + std::to_string(c.beta)
            + " bias=" + std::to_string(static_cast<int>(c.bias_dt)) + " relu="
            + std::to_string(c.relu) + " binary=" + std::to_string(c.binary)
            + " dst=" + std::to_string(static_cast<int>(c.dst_dt)) + " threads="
            + std::to_string(c.threads) + " ldc=" + std::to_string(c.ldc());
}

/// The dst values a run must not write: f16 0x5555, f32 -7.
constexpr uint16_t kUnwrittenF16 = 0x5555;
constexpr float kUnwrittenF32 = -7.0f;

struct f16_operands_t {
    std::vector<uint16_t> src, wei, bias16, bin16;
    std::vector<float> bias32, bin32;
    /// The dst before the call, of dst_dt, rows ldc apart.
    buffer_t dst0;
};

size_t dst_elems(const f16_call_t &c) {
    return static_cast<size_t>(c.M) * c.ldc();
}

f16_operands_t operands_of(const f16_call_t &c, uint32_t seed) {
    f16_operands_t o;
    o.src = random_f16(
            static_cast<size_t>(c.transA ? c.K : c.M) * c.lda(), seed);
    o.wei = random_f16(
            static_cast<size_t>(c.transB ? c.N : c.K) * c.ldb(), seed + 1);
    if (c.bias_dt == kF32) o.bias32 = random_floats(c.N, seed + 2);
    if (c.bias_dt == kF16) o.bias16 = random_f16(c.N, seed + 2);
    if (c.bias_dt == kBf16) {
        const buffer_t b = nt::random_buffer(kBf16, c.N, seed + 2, -1, 1);
        o.bias16.assign(b.as<uint16_t>(), b.as<uint16_t>() + c.N);
    }
    if (c.binary == 1)
        o.bin16 = random_f16(static_cast<size_t>(c.M) * c.N, seed + 3);
    if (c.binary == 2) o.bin32 = random_floats(c.N, seed + 3, 0.5f, 1.5f);
    const size_t n = dst_elems(c);
    o.dst0 = nt::make_buffer(n * nt::dt_bytes(c.dst_dt));
    const std::vector<float> old = random_floats(n, seed + 4);
    for (size_t i = 0; i < n; ++i) {
        const bool inside = static_cast<int>(i % c.ldc()) < c.N;
        if (c.dst_dt == kF16)
            o.dst0.as<uint16_t>()[i] = inside ? f16_of(old[i]) : kUnwrittenF16;
        else
            o.dst0.as<float>()[i] = inside ? old[i] : kUnwrittenF32;
    }
    return o;
}

float src_at(const f16_call_t &c, const f16_operands_t &o, int m, int k) {
    return f32_of(c.transA ? o.src[static_cast<size_t>(k) * c.lda() + m]
                           : o.src[static_cast<size_t>(m) * c.lda() + k]);
}

float wei_at(const f16_call_t &c, const f16_operands_t &o, int k, int n) {
    return f32_of(c.transB ? o.wei[static_cast<size_t>(n) * c.ldb() + k]
                           : o.wei[static_cast<size_t>(k) * c.ldb() + n]);
}

float bias_at(const f16_call_t &c, const f16_operands_t &o, int n) {
    if (c.bias_dt == kF32) return o.bias32[n];
    if (c.bias_dt == kF16) return f32_of(o.bias16[n]);
    return bf16_value(o.bias16[n]);
}

float dst0_at(const f16_call_t &c, const f16_operands_t &o, size_t i) {
    return c.dst_dt == kF16 ? f32_of(o.dst0.as<uint16_t>()[i])
                            : o.dst0.as<float>()[i];
}

/// The products of the inputs added as the kernels of f16 calls add them
/// (f16_dot: blocked on the F16_PH kernels, else the widened products in K
/// order, one f32 fused multiply-add each), from +0.0; then alpha (one
/// product), beta times the old dst (one product, one sum), the bias (one
/// sum), relu and the binary operand (one sum or product).  As f32, rows ldc
/// apart; past N the old dst.
std::vector<float> model_output(const f16_call_t &c, const f16_operands_t &o) {
    std::vector<float> out(dst_elems(c));
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = dst0_at(c, o, i);
    const bool blocked = f16_ph_calls();
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n) {
            float acc = f16_dot(c.K, [&](int k) { return src_at(c, o, m, k); },
                    [&](int k) { return wei_at(c, o, k, n); }, blocked);
            const size_t at = static_cast<size_t>(m) * c.ldc() + n;
            if (c.alpha != 1.0f) acc = acc * c.alpha;
            if (c.beta != 0.0f) acc = acc + dst0_at(c, o, at) * c.beta;
            if (c.bias_dt != kNone) acc = acc + bias_at(c, o, n);
            if (c.relu) acc = acc > 0.0f ? acc : 0.0f;
            if (c.binary == 1)
                acc = acc + f32_of(o.bin16[static_cast<size_t>(m) * c.N + n]);
            if (c.binary == 2) acc = acc * o.bin32[n];
            out[at] = acc;
        }
    return out;
}

mm::matmul_params params_of(const f16_call_t &c, const f16_operands_t &o) {
    mm::matmul_params p;
    p.dtypes.src = p.dtypes.wei = kF16;
    p.dtypes.dst = c.dst_dt;
    p.dtypes.bias = c.bias_dt;
    p.num_threads = c.threads;
    if (c.relu) {
        mm::matmul_post_op po;
        po.po_type = post_op_type_t::relu;
        po.alpha = 0.0f;
        p.postop_.push_back(po);
    }
    if (c.binary != 0) {
        mm::matmul_post_op po;
        po.po_type = c.binary == 1 ? post_op_type_t::binary_add
                                   : post_op_type_t::binary_mul;
        po.buff = c.binary == 1
                ? static_cast<void *>(const_cast<uint16_t *>(o.bin16.data()))
                : static_cast<void *>(const_cast<float *>(o.bin32.data()));
        po.dtype = c.binary == 1 ? kF16 : kF32;
        po.dims = c.binary == 1 ? std::vector<int64_t> {c.M, c.N}
                                : std::vector<int64_t> {1, c.N};
        po.leading_dim = c.N;
        p.postop_.push_back(po);
    }
    return p;
}

const void *bias_ptr(const f16_call_t &c, const f16_operands_t &o) {
    if (c.bias_dt == kF32) return o.bias32.data();
    if (c.bias_dt == kNone) return nullptr;
    return o.bias16.data();
}

nat::GemmDescriptor desc_of(const f16_call_t &c, const f16_operands_t &o) {
    nat::GemmDescriptor d;
    d.M = c.M;
    d.N = c.N;
    d.K = c.K;
    d.lda = c.lda();
    d.ldb = c.ldb();
    d.ldc = c.ldc();
    d.transA = c.transA;
    d.transB = c.transB;
    d.alpha = c.alpha;
    d.beta = c.beta;
    d.src_dt = d.wei_dt = kF16;
    d.dst_dt = c.dst_dt;
    d.bias_dt = c.bias_dt;
    d.bias = bias_ptr(c, o);
    d.is_weights_const = false;
    d.num_threads = c.threads;
    return d;
}

/// The call on the native BRGEMM, the weight packed for it alone, into a
/// copy of dst0.
buffer_t run_native(const f16_call_t &c, const f16_operands_t &o) {
    buffer_t out = nt::make_buffer(o.dst0.bytes);
    std::memcpy(out.get(), o.dst0.get(), o.dst0.bytes);
    nat::native_weight_ref_t ref;
    ref.weight = o.wei.data();
    ref.pack_nr = kNr;
    ref.weight_cache_mode = 0;
    const mm::matmul_params p = params_of(c, o);
    EXPECT_EQ(nat::native_brgemm_execute(desc_of(c, o), ref, o.src.data(),
                      out.get(), bias_ptr(c, o), p),
            status_t::success)
            << str(c);
    return out;
}

/// The dst of the model stored in dst_dt, rounded to nearest even for f16,
/// equal in bits to got, or the first element that differs.
::testing::AssertionResult same_bits(const f16_call_t &c, const buffer_t &got,
        const std::vector<float> &want) {
    for (size_t i = 0; i < want.size(); ++i) {
        const bool equal = c.dst_dt == kF16
                ? got.as<uint16_t>()[i] == f16_of(want[i])
                : bits_of(got.as<float>()[i]) == bits_of(want[i]);
        if (!equal) {
            const float g = c.dst_dt == kF16 ? f32_of(got.as<uint16_t>()[i])
                                             : got.as<float>()[i];
            return ::testing::AssertionFailure()
                    << "element " << i << " (row " << i / c.ldc() << ", col "
                    << i % c.ldc() << "): " << g << " for " << want[i];
        }
    }
    return ::testing::AssertionSuccess();
}

std::vector<f16_call_t> bit_calls() {
    std::vector<f16_call_t> v;
    const int shapes[][3] = {{1, 32, 64}, {1, 100, 300}, {5, 96, 1}, {6, 64, 7},
            {7, 160, 300}, {13, 33, 1031}, {100, 256, 512}, {300, 96, 70},
            {40, 64, 9000}};
    int variant = 0;
    for (const auto &s : shapes)
        for (const bool transB : {false, true})
            for (const bool transA : {false, true})
                for (const data_type_t dst : {kF16, kF32}) {
                    f16_call_t c;
                    c.M = s[0];
                    c.N = s[1];
                    c.K = s[2];
                    c.transA = transA;
                    c.transB = transB;
                    c.dst_dt = dst;
                    c.threads = variant % 3 == 0 ? 1 : 4;
                    // The store only, then alpha, beta, each bias type,
                    // relu and the binary operands in turn.
                    switch (variant++ % 6) {
                        case 0: break;
                        case 1:
                            c.alpha = 1.5f;
                            c.bias_dt = kF16;
                            break;
                        case 2:
                            c.beta = 0.5f;
                            c.bias_dt = kF32;
                            c.relu = true;
                            break;
                        case 3:
                            c.bias_dt = kBf16;
                            c.binary = 1;
                            break;
                        case 4:
                            c.alpha = -0.75f;
                            c.beta = 1.0f;
                            c.binary = 2;
                            break;
                        default: c.ldc_pad = 7; break;
                    }
                    v.push_back(c);
                }
    return v;
}

// The native BRGEMM adds each output's products of the widened f16 inputs
// in K order from zero, whatever its tiles, K passes and threads, also when
// a call left with only the store accumulates in an f32 destination: the
// bits of the scalar model, with alpha, beta, the bias of each type, relu
// and the binary operands after the sum, an f16 dst rounded to nearest
// even, and the padding of dst rows untouched.
TEST(NativeBrgemmF16, OutputsAddWidenedProductsInKOrder) {
    SKIP_WITHOUT_NATIVE_ISA();
    uint32_t seed = 500;
    for (const f16_call_t &c : bit_calls()) {
        SCOPED_TRACE(str(c));
        const f16_operands_t o = operands_of(c, seed += 7);
        EXPECT_TRUE(same_bits(c, run_native(c, o), model_output(c, o)));
    }
}

// Every N from 1 to 130 (each tail of the 32-column blocks, the weight
// padded with zero columns) at one, two and a few rows, into f16 and f32:
// the bits of the model, nothing past N written.
TEST(NativeBrgemmF16, EveryNTailMatchesTheModel) {
    SKIP_WITHOUT_NATIVE_ISA();
    uint32_t seed = 900;
    int failures = 0;
    for (int N = 1; N <= 130; ++N)
        for (const int M : {1, 2, 13}) {
            f16_call_t c;
            c.M = M;
            c.N = N;
            c.K = 37 + N % 5;
            c.transB = N % 2 == 0;
            c.dst_dt = N % 3 == 0 ? kF32 : kF16;
            c.bias_dt = N % 4 == 0 ? kF16 : kNone;
            c.threads = N % 7 == 0 ? 1 : 4;
            c.ldc_pad = N % 5;
            const f16_operands_t o = operands_of(c, seed += 3);
            const ::testing::AssertionResult r
                    = same_bits(c, run_native(c, o), model_output(c, o));
            if (!r && failures++ < 10) ADD_FAILURE() << str(c) << ": " << r;
        }
    EXPECT_EQ(failures, 0);
}

/// Calls of K passes under the plans of plan_pins: K past the passes, one
/// row to a few hundred, transposed sources, f16 and f32 dsts, a bias.
std::vector<f16_call_t> pass_calls() {
    std::vector<f16_call_t> v;
    const int shapes[][3] = {{1, 100, 1300}, {9, 33, 777}, {37, 96, 1031},
            {130, 64, 2100}, {300, 160, 900}};
    int i = 0;
    for (const auto &s : shapes)
        for (const int threads : {1, 4, 13}) {
            f16_call_t c;
            c.M = s[0];
            c.N = s[1];
            c.K = s[2];
            c.transA = i % 2 == 1;
            c.transB = i % 3 == 0;
            c.dst_dt = i % 4 < 2 ? kF16 : kF32;
            c.bias_dt = i % 5 == 0 ? kF32 : kNone;
            c.threads = threads;
            ++i;
            v.push_back(c);
        }
    return v;
}

/// ZENDNNL_MATMUL_NATIVE_CONFIG clauses for f16 calls: K passes that are
/// whole blocks of kNativeF16AccumBlockK and passes that are not, row calls
/// and tiles of other widths (the F16_PH tiles of 64 columns from one row),
/// M blocks, the n_inner order and the blocks schedule.
const char *const kPlanPins[] = {"dtype=f16:kc=200", "dtype=f16:kc=2",
        "dtype=f16:kc=700,mr=5,nr=16", "dtype=f16:kc=256,mb=24,mr=7",
        "dtype=f16:f16_wide_min_m=1,kc=700",
        "dtype=f16:schedule=blocks,unit_tiles=2,kc=384,mb=24",
        "dtype=f16:traversal=n_inner,mb=36,kc=640"};

// The bits of an F16 call do not depend on its plan: under pinned plans of
// K passes (whole blocks or not, which an F16_PH plan rounds down to whole
// blocks), row calls, tile widths, M blocks, orders and schedules, and on
// 1, 4 and 13 threads, every call gives the bits of the model, which no
// plan enters.  The pins hold: the plans take them from the override, and
// an F16_PH plan's K pass is whole blocks.
TEST(NativeBrgemmF16, BitsAreTheModelsWhateverThePlan) {
    SKIP_WITHOUT_NATIVE_ISA();
    const std::vector<f16_call_t> calls = pass_calls();
    for (const char *pin : kPlanPins) {
        nt::expect_in_child(
                {{"ZENDNNL_MATMUL_NATIVE_CONFIG", pin},
                        {"ZENDNNL_MATMUL_NATIVE_TUNING_FILE", nullptr}},
                [&] {
            int bad = 0;
            uint32_t seed = 3000;
            for (const f16_call_t &c : calls) {
                const f16_operands_t o = operands_of(c, seed += 5);
                const ::testing::AssertionResult r
                        = same_bits(c, run_native(c, o), model_output(c, o));
                if (!r) {
                    std::fprintf(stderr, "%s under %s: %s\n", str(c).c_str(),
                            pin, r.message());
                    ++bad;
                }
                const nat::native_plan_t p = nat::native_planner().plan(
                        problem(c.M, nat::native_padded_n(c.N, kNr), c.K,
                                c.threads, kF16, c.dst_dt));
                const bool whole = p.kc >= c.K
                        || (p.kc >= nat::kNativeF16AccumBlockK
                                && p.kc % nat::kNativeF16AccumBlockK == 0);
                if (nt::top_source(p) != nat::native_plan_source_t::override
                        || (p.kind == nat::native_kernel_kind_t::f16_ph
                                && !whole)) {
                    std::fprintf(stderr, "%s under %s: plan kc %d source %s\n",
                            str(c).c_str(), pin, p.kc,
                            nat::native_plan_source_names(p).c_str());
                    ++bad;
                }
            }
            return bad + (::testing::Test::HasFailure() ? 1 : 0);
        }, pin);
    }
}

/// Calls of N whose padded width is whole tiles of 64 or 128 columns (with
/// tails), M 2 to 300, transposed sources, f16 and f32 dsts, a bias.
std::vector<f16_call_t> wide_calls() {
    std::vector<f16_call_t> v;
    const int shapes[][3] = {{2, 64, 300}, {9, 100, 777}, {13, 128, 1300},
            {70, 250, 600}, {300, 1000, 513}};
    int i = 0;
    for (const auto &s : shapes)
        for (const int threads : {1, 4}) {
            f16_call_t c;
            c.M = s[0];
            c.N = s[1];
            c.K = s[2];
            c.transA = i % 2 == 1;
            c.transB = i % 3 == 0;
            c.dst_dt = i % 4 < 2 ? kF16 : kF32;
            c.bias_dt = i % 5 == 0 ? kF32 : kNone;
            c.threads = threads;
            ++i;
            v.push_back(c);
        }
    return v;
}

// Tiles of 64 and 128 columns give the model's bits: 32-column slabs of the
// F16 kernels, or the F16_PH tile of 64 and two of them; with K passes, rows
// per call, M blocks and the blocks schedule.  Each pin runs in a child
// process on the calls whose padded N is whole tiles of its width.
TEST(NativeBrgemmF16, WideTilesGiveTheModelsBits) {
    SKIP_WITHOUT_NATIVE_ISA();
    const struct {
        const char *pin;
        int nr;
    } pins[] = {{"dtype=f16:nr=64", 64}, {"dtype=f16:nr=128", 128},
            {"dtype=f16:nr=128,kc=512,mr=5", 128},
            {"dtype=f16:nr=64,schedule=blocks,nb=256,kc=256,mb=24", 64}};
    const std::vector<f16_call_t> calls = wide_calls();
    for (const auto &pin : pins) {
        nt::expect_in_child(
                {{"ZENDNNL_MATMUL_NATIVE_CONFIG", pin.pin},
                        {"ZENDNNL_MATMUL_NATIVE_TUNING_FILE", nullptr}},
                [&] {
            int bad = 0, taken = 0;
            uint32_t seed = 3500;
            for (const f16_call_t &c : calls) {
                const int padded = nat::native_padded_n(c.N, kNr);
                if (padded % pin.nr != 0) continue;
                const f16_operands_t o = operands_of(c, seed += 5);
                const ::testing::AssertionResult r
                        = same_bits(c, run_native(c, o), model_output(c, o));
                const nat::native_plan_t p = nat::native_planner().plan(
                        problem(c.M, padded, c.K, c.threads, kF16, c.dst_dt));
                if (!r || p.nr != pin.nr
                        || nt::top_source(p)
                                != nat::native_plan_source_t::override) {
                    std::fprintf(stderr, "%s under %s: plan nr %d, %s\n",
                            str(c).c_str(), pin.pin, p.nr, r.message());
                    ++bad;
                }
                ++taken;
            }
            if (taken < 4) {
                std::fprintf(stderr, "%s took %d calls\n", pin.pin, taken);
                ++bad;
            }
            return bad + (::testing::Test::HasFailure() ? 1 : 0);
        }, pin.pin);
    }
}

// The native BRGEMM turns an F16 call with a dst other than f16 / f32 away
// before any work: unimplemented, and dst untouched.
TEST(NativeBrgemmF16, OtherDstTypesAreTurnedAway) {
    SKIP_WITHOUT_NATIVE_ISA();
    f16_call_t c;
    c.M = 4;
    c.N = 64;
    c.K = 32;
    const f16_operands_t o = operands_of(c, 41);
    for (const data_type_t dst :
            {kBf16, data_type_t::s8, data_type_t::u8, data_type_t::s32}) {
        nat::GemmDescriptor d = desc_of(c, o);
        d.dst_dt = dst;
        mm::matmul_params p = params_of(c, o);
        p.dtypes.dst = dst;
        nat::native_weight_ref_t ref;
        ref.weight = o.wei.data();
        ref.pack_nr = kNr;
        ref.weight_cache_mode = 0;
        std::vector<float> out(static_cast<size_t>(c.M) * c.N, kUnwrittenF32);
        const char *why = nullptr;
        EXPECT_EQ(nat::native_brgemm_execute(
                          d, ref, o.src.data(), out.data(), nullptr, p, &why),
                status_t::unimplemented);
        EXPECT_STREQ(why, "an f16 call into a dst other than f16 or f32");
        EXPECT_TRUE(std::all_of(out.begin(), out.end(),
                [](float v) { return v == kUnwrittenF32; }));
    }
}

#if defined(__linux__)
/// An anonymous mapping of bytes that reads zero and holds no memory but
/// the pages of [from, bytes), which it can write.
struct sparse_map_t {
    uint8_t *base = nullptr;
    size_t bytes = 0;

    sparse_map_t(size_t n, size_t from) {
        const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        bytes = (n + page - 1) / page * page;
        void *p = mmap(nullptr, bytes, PROT_READ,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) return;
        base = static_cast<uint8_t *>(p);
        const size_t lo = from / page * page;
        if (mprotect(base + lo, bytes - lo, PROT_READ | PROT_WRITE) != 0) {
            munmap(base, bytes);
            base = nullptr;
        }
    }
    ~sparse_map_t() {
        if (base != nullptr) munmap(base, bytes);
    }
    sparse_map_t(const sparse_map_t &) = delete;
    sparse_map_t &operator=(const sparse_map_t &) = delete;
};

// The widening of an f16 source of M 2147483640 rows and K 1 in blocks of
// 64 rows (the plans' copy_block): the last block starts at 2147483584,
// which plus 64 passes the int range, as does the start of a transposed
// source's last step of 16 lanes plus 16.  The block ends at M, and that
// step takes the 8 rows left: over mappings of the source and the widened
// rows that hold only the pages the block reads and writes, every row of
// the block gets its value widened, and the rows before it in its page
// keep theirs.
TEST(NativeBrgemmF16, TheLastBlockOfAWidenedSourceEndsAtM) {
    SKIP_WITHOUT_NATIVE_ISA();
    const int M = 2147483640, rows = 64;
    const int b = (M - 1) / rows, m0 = b * rows;
    ASSERT_EQ(m0, 2147483584);
    const sparse_map_t src(
            static_cast<size_t>(M) * 2, static_cast<size_t>(m0) * 2);
    const sparse_map_t dst(
            static_cast<size_t>(M) * 4, static_cast<size_t>(m0) * 4);
    if (src.base == nullptr || dst.base == nullptr)
        GTEST_SKIP() << "no address space for the source";
    uint16_t *const s = reinterpret_cast<uint16_t *>(src.base);
    float *const d = reinterpret_cast<float *>(dst.base);
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t first = static_cast<size_t>(m0) * 4 / page * page / 4;
    for (int m = m0; m < M; ++m)
        s[m] = f16_of(static_cast<float>(m - m0) + 0.5f);
    for (const bool trans : {false, true}) {
        SCOPED_TRACE(trans ? "transposed" : "row-major");
        for (size_t m = first; m < static_cast<size_t>(M); ++m)
            d[m] = kUnwrittenF32;
        // [M, 1] rows 1 apart, or [1, M]: element (m, 0) is s[m] either way.
        nat::native_widen_f16_block(
                s, trans ? M : 1, trans, M, 1, d, 1, rows, 64, b);
        int bad = 0;
        for (size_t m = first; m < static_cast<size_t>(M); ++m) {
            const float want = m < static_cast<size_t>(m0) ? kUnwrittenF32
                                                           : f32_of(s[m]);
            bad += bits_of(d[m]) != bits_of(want);
        }
        EXPECT_EQ(bad, 0);
    }
}
#endif

// ── matmul_direct under ALGO 10 ──────────────────────────────────────────

/// An F16 call of matmul_direct on a weight of its own, and its operands.
struct direct_t {
    f16_call_t shape;
    f16_operands_t o;
    buffer_t wei, dst;
    nt::call_t call;
};

direct_t make_direct(const f16_call_t &s, uint32_t seed, int32_t wc = 1) {
    direct_t d;
    d.shape = s;
    d.o = operands_of(s, seed);
    d.wei = nt::kept_until_exit(nt::make_buffer(d.o.wei.size() * 2));
    std::memcpy(d.wei.get(), d.o.wei.data(), d.o.wei.size() * 2);
    d.dst = nt::make_buffer(d.o.dst0.bytes);
    std::memcpy(d.dst.get(), d.o.dst0.get(), d.o.dst0.bytes);
    nt::call_t &c = d.call;
    c.M = s.M;
    c.N = s.N;
    c.K = s.K;
    c.transA = s.transA;
    c.transB = s.transB;
    c.lda = s.lda();
    c.ldb = s.ldb();
    c.ldc = s.ldc();
    c.alpha = s.alpha;
    c.beta = s.beta;
    c.src = d.o.src.data();
    c.wei = d.wei.get();
    c.bias = bias_ptr(s, d.o);
    c.dst = d.dst.get();
    c.params = params_of(s, d.o);
    c.params.weight_cache_type = wc;
    return d;
}

/// d's call with a tight ldb (an image fits only a contiguous weight).
direct_t make_contiguous(int M, int N, int K, bool transB, data_type_t dst,
        uint32_t seed, int32_t wc) {
    f16_call_t s;
    s.M = M;
    s.N = N;
    s.K = K;
    s.transB = transB;
    s.dst_dt = dst;
    s.threads = 4;
    direct_t d = make_direct(s, seed, wc);
    d.call.ldb = transB ? K : N;
    // The model reads the weight at the call's tight ldb.
    std::vector<uint16_t> tight(static_cast<size_t>(K) * N);
    for (int r = 0; r < (transB ? N : K); ++r)
        for (int e = 0; e < (transB ? K : N); ++e)
            tight[static_cast<size_t>(r) * d.call.ldb + e]
                    = d.o.wei[static_cast<size_t>(r) * s.ldb() + e];
    d.wei = nt::kept_until_exit(nt::make_buffer(tight.size() * 2));
    std::memcpy(d.wei.get(), tight.data(), tight.size() * 2);
    d.call.wei = d.wei.get();
    return d;
}

/// The model of d's call over the weight its call reads.
std::vector<float> model_of(const direct_t &d) {
    f16_call_t s = d.shape;
    f16_operands_t o = d.o;
    if (d.call.ldb != s.ldb()) {
        // Back to the model's padded rows.
        const int rows = s.transB ? s.N : s.K, row = s.transB ? s.K : s.N;
        const uint16_t *w = d.wei.as<uint16_t>();
        std::vector<uint16_t> raw(o.wei.size());
        for (int r = 0; r < rows; ++r)
            for (int e = 0; e < row; ++e)
                raw[static_cast<size_t>(r) * s.ldb() + e]
                        = w[static_cast<size_t>(r) * d.call.ldb + e];
        o.wei = raw;
    }
    return model_output(s, o);
}

std::vector<f16_call_t> route_calls() {
    std::vector<f16_call_t> v;
    const int shapes[][3] = {{1, 64, 128}, {1, 4096, 512}, {33, 96, 1031},
            {8, 4096, 512}, {8, 48, 64}, {1, 1, 200}, {33, 100, 1031},
            {130, 256, 2048}, {300, 96, 700}};
    int i = 0;
    for (const auto &s : shapes) {
        f16_call_t c;
        c.M = s[0];
        c.N = s[1];
        c.K = s[2];
        c.transA = i % 3 == 1;
        c.transB = i % 2 == 0;
        c.dst_dt = i % 3 == 2 ? kF32 : kF16;
        c.bias_dt = i % 4 == 1 ? kF16 : (i % 4 == 3 ? kF32 : kNone);
        c.threads = 4;
        ++i;
        v.push_back(c);
    }
    return v;
}

// Under ALGO 10, and per call with lowoha_algo = native_brgemm, F16 calls
// run on the native BRGEMM, on a CPU with or without AVX512-FP16, with the
// bits of the scalar model: a weight in whole blocks and one padded past
// N, one row, a few and a few hundred, transposed sources, f16 and f32
// dsts, f16 and f32 biases.
TEST(NativeBrgemmF16Route, Algo10AndPerCallServeF16Calls) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    // The layout check needs a cached layout whatever the process's mode.
    const WeightCacheGuard cache(2);
    for (const bool per_call : {false, true}) {
        SCOPED_TRACE(per_call ? "lowoha_algo=native_brgemm" : "ALGO 10");
        const nt::algo_pin_t pin(
                per_call ? matmul_algo_t::none : matmul_algo_t::native_brgemm);
        uint32_t seed = 70;
        for (const f16_call_t &s : route_calls()) {
            SCOPED_TRACE(str(s));
            direct_t d = make_direct(s, seed += 3);
            if (per_call)
                d.call.params.lowoha_algo = matmul_algo_t::native_brgemm;
            ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
            EXPECT_TRUE(same_bits(s, d.dst, model_of(d)));
            EXPECT_NE(nt::native_layouts(d.wei.get()), 0u);
        }
    }
    mm::clear_matmul_weight_caches();
}

// The gated activations of matmul_direct (silu / gelu over [gate | up]
// halves, swiglu over interleaved pairs) and the eltwise post-ops gelu_erf,
// sigmoid and tanh run on the F16 tiles within the band of the f32 model,
// into f16 (one rounding step) and f32.
TEST(NativeBrgemmF16Route, GatedActivationsAndPostOpsWithinTheBand) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    uint32_t seed = 1300;
    for (const gated_t act : {gated_t::silu_and_mul, gated_t::gelu_and_mul,
                 gated_t::swiglu_oai_mul, gated_t::none})
        for (const post_op_type_t eltwise :
                {post_op_type_t::none, post_op_type_t::gelu_erf,
                        post_op_type_t::sigmoid, post_op_type_t::tanh})
            for (const data_type_t dst : {kF16, kF32}) {
                if (act == gated_t::none && eltwise == post_op_type_t::none)
                    continue;
                for (const int N : {256, 96, 70}) {
                    SCOPED_TRACE("act " + std::to_string(static_cast<int>(act))
                            + " eltwise "
                            + std::to_string(static_cast<int>(eltwise))
                            + " dst " + std::to_string(static_cast<int>(dst))
                            + " N " + std::to_string(N));
                    f16_call_t s;
                    s.M = 9;
                    s.N = N;
                    s.K = 200;
                    s.transB = N != 96;
                    s.bias_dt = N == 96 ? kF16 : kNone;
                    s.dst_dt = dst;
                    s.threads = 4;
                    const int out_cols = act == gated_t::none ? N : N / 2;
                    s.ldc_pad = out_cols - N;
                    direct_t d = make_direct(s, seed += 5);
                    d.call.gated = act;
                    d.call.ldc = out_cols;
                    if (eltwise != post_op_type_t::none) {
                        mm::matmul_post_op po;
                        po.po_type = eltwise;
                        d.call.params.postop_.push_back(po);
                    }
                    ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
                    // The model of the pre-activation, then the post-op and
                    // the activation in f32.
                    f16_call_t pre_shape = s;
                    pre_shape.ldc_pad = 0;
                    f16_operands_t po = d.o;
                    po.dst0 = nt::make_buffer(
                            static_cast<size_t>(s.M) * s.N * 4);
                    pre_shape.dst_dt = kF32;
                    std::vector<float> pre = model_output(pre_shape, po);
                    for (float &v : pre) {
                        if (eltwise == post_op_type_t::gelu_erf)
                            v = 0.5f * v
                                    * (1.0f + std::erf(v / std::sqrt(2.0f)));
                        else if (eltwise == post_op_type_t::sigmoid)
                            v = 1.0f / (1.0f + std::exp(-v));
                        else if (eltwise == post_op_type_t::tanh)
                            v = std::tanh(v);
                    }
                    std::vector<float> want(
                            static_cast<size_t>(s.M) * out_cols);
                    if (act == gated_t::none)
                        want = pre;
                    else
                        nt::gated_act_ref(
                                pre.data(), s.M, s.N, act, want.data());
                    int bad = 0;
                    for (int m = 0; m < s.M; ++m)
                        for (int n = 0; n < out_cols; ++n) {
                            const size_t i
                                    = static_cast<size_t>(m) * out_cols + n;
                            const float got = dst == kF16
                                    ? f32_of(d.dst.as<uint16_t>()[i])
                                    : d.dst.as<float>()[i];
                            const float w = want[i];
                            const float tol = dst == kF16
                                    ? std::fabs(w) / 1024.0f + 1e-4f
                                    : std::fabs(w) * 1e-5f + 1e-5f;
                            bad += !(std::fabs(got - w) <= tol);
                        }
                    EXPECT_EQ(bad, 0);
                }
            }
    mm::clear_matmul_weight_caches();
}

// The weight cache serves an f16 weight as a bf16 one: mode 1 keeps one
// layout out of place and the raw weight as it was; under ALGO 10 mode 2
// rewrites a contiguous weight of whole blocks in place, and caches a
// weight padded past N out of place; mode 0 keeps none.  Every call gives
// the model's bits.
TEST(NativeBrgemmF16Route, WeightCacheModesKeepTheBits) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    uint32_t seed = 90;
    for (const int32_t wc : {0, 1, 2})
        for (const int N : {128, 100})
            for (const bool transB : {false, true}) {
                SCOPED_TRACE("weight cache " + std::to_string(wc) + " N "
                        + std::to_string(N) + " transB "
                        + std::to_string(transB));
                WeightCacheGuard cache(wc);
                direct_t d = make_contiguous(
                        4, N, 256, transB, kF16, seed += 3, wc);
                const std::vector<float> want = model_of(d);
                const std::vector<uint8_t> raw(
                        d.wei.as<uint8_t>(), d.wei.as<uint8_t>() + d.wei.bytes);
                for (int call = 0; call < 3; ++call) {
                    std::memcpy(d.dst.get(), d.o.dst0.get(), d.o.dst0.bytes);
                    ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
                    EXPECT_TRUE(same_bits(d.shape, d.dst, want)) << call;
                }
                const bool in_place = wc == 2 && N % kNr == 0;
                EXPECT_EQ(nat::native_weight_packed_in_place(
                                  d.wei.get(), d.wei.bytes),
                        in_place);
                EXPECT_EQ(
                        std::memcmp(raw.data(), d.wei.get(), d.wei.bytes) == 0,
                        !in_place)
                        << "the raw weight was rewritten";
                EXPECT_EQ(nt::native_layouts(d.wei.get()), wc == 0 ? 0u : 1u);
            }
    mm::clear_matmul_weight_caches();
}

// A non-transposed f16 weight of N and ldb 32 is its own layout: under ALGO
// 10 weight cache mode 2 does not rewrite it in place, caches its layout out
// of place and keeps the buffer as it was, and after a clear and a reload of
// the raw weight the next call gives the model's bits.
TEST(NativeBrgemmF16Route, AWeightThatIsItsOwnLayoutIsNotRewritten) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    WeightCacheGuard cache(2);
    direct_t d = make_contiguous(4, kNr, 256, false, kF32, 1450, 2);
    const std::vector<float> want = model_of(d);
    const std::vector<uint8_t> raw(
            d.wei.as<uint8_t>(), d.wei.as<uint8_t>() + d.wei.bytes);
    std::memcpy(d.dst.get(), d.o.dst0.get(), d.o.dst0.bytes);
    ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
    EXPECT_TRUE(same_bits(d.shape, d.dst, want));
    EXPECT_FALSE(nat::native_weight_packed_in_place(d.wei.get(), d.wei.bytes));
    EXPECT_EQ(std::memcmp(raw.data(), d.wei.get(), d.wei.bytes), 0);
    EXPECT_EQ(nt::native_layouts(d.wei.get()), 1u);
    mm::clear_matmul_weight_caches();
    std::memcpy(d.wei.get(), raw.data(), raw.size());
    std::memcpy(d.dst.get(), d.o.dst0.get(), d.o.dst0.bytes);
    ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
    EXPECT_TRUE(same_bits(d.shape, d.dst, want));
    mm::clear_matmul_weight_caches();
}

// A weight rewritten in place for plain calls serves a later silu_and_mul
// call on the same buffer from the f16 image (the layout depends on the
// weight alone): the bits of the same call over a pack made for it.
TEST(NativeBrgemmF16Route, InPlaceImageServesAGatedCall) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    WeightCacheGuard cache(2);
    direct_t d = make_contiguous(6, 128, 192, true, kF16, 1500, 2);
    ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
    ASSERT_TRUE(nat::native_weight_packed_in_place(d.wei.get(), d.wei.bytes));
    // The same call, gated, on a raw copy at weight cache mode 0.
    direct_t fresh = make_contiguous(6, 128, 192, true, kF16, 1500, 0);
    fresh.call.gated = gated_t::silu_and_mul;
    fresh.call.ldc = 64;
    fresh.call.params.weight_cache_type = 0;
    std::vector<uint16_t> want(static_cast<size_t>(6) * 64, kUnwrittenF16);
    fresh.call.dst = want.data();
    ASSERT_EQ(nt::run_matmul(fresh.call), status_t::success);
    d.call.gated = gated_t::silu_and_mul;
    d.call.ldc = 64;
    std::vector<uint16_t> got(want.size(), kUnwrittenF16);
    d.call.dst = got.data();
    ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
    EXPECT_EQ(got, want);
    mm::clear_matmul_weight_caches();
}

// A call on a narrower view of a buffer that holds an f16 image, its first
// N / 2 columns with ldb N, packs its layout out of place from the raw
// weight the image unpacks to.  That rebuild reads the whole image, so the
// view must state the buffer as its capacity: without it the call fails
// with dst untouched; with it the call gives the scalar model's bits of
// the view over the raw weight, and the image stays.
TEST(NativeBrgemmF16Route, AViewOfAnImageIsPackedFromTheWeightItHolds) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    WeightCacheGuard cache(2);
    const int M = 4, K = 64, N = 64;
    direct_t whole = make_contiguous(M, N, K, false, kF32, 1550, 2);
    const std::vector<uint16_t> raw(whole.wei.as<uint16_t>(),
            whole.wei.as<uint16_t>() + static_cast<size_t>(K) * N);
    ASSERT_EQ(nt::run_matmul(whole.call), status_t::success);
    ASSERT_TRUE(nat::native_weight_packed_in_place(
            whole.wei.get(), whole.wei.bytes));
    const std::vector<uint8_t> image(
            whole.wei.as<uint8_t>(), whole.wei.as<uint8_t>() + whole.wei.bytes);
    f16_call_t s;
    s.M = M;
    s.N = N / 2;
    s.K = K;
    s.dst_dt = kF32;
    s.threads = 4;
    direct_t view = make_direct(s, 1551, 2);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < s.N; ++n)
            view.o.wei[static_cast<size_t>(k) * s.ldb() + n]
                    = raw[static_cast<size_t>(k) * N + n];
    const std::vector<float> want = model_output(s, view.o);
    view.call.wei = whole.wei.get();
    view.call.ldb = N;
    EXPECT_EQ(nt::run_matmul(view.call), status_t::failure)
            << "a rebuild past the bytes the view reads";
    EXPECT_EQ(std::memcmp(view.dst.get(), view.o.dst0.get(), view.o.dst0.bytes),
            0)
            << "dst written";
    view.call.params.wei_buffer_capacity_bytes = whole.wei.bytes;
    ASSERT_EQ(nt::run_matmul(view.call), status_t::success);
    EXPECT_TRUE(same_bits(s, view.dst, want));
    EXPECT_EQ(std::memcmp(whole.wei.get(), image.data(), image.size()), 0)
            << "the image was rewritten";
    EXPECT_EQ(nt::native_layouts(whole.wei.get()), 2u)
            << "the image and the view's layout";
    mm::clear_matmul_weight_caches();
}

// A call on the last rows of a weight, whose layout a call cached at weight
// cache mode 1 before a mode 2 call rewrote the whole weight in place, is
// served that layout at modes 1 and 2: the scalar model's bits of the view
// over its raw weight.  A mode 0 call on it is refused, dst untouched: the
// image holds no raw weight for it.
TEST(NativeBrgemmF16Route, AViewInsideAnImageIsServedItsCachedLayout) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    WeightCacheGuard cache(2);
    const int M = 4, K = 128, N = 64;
    direct_t whole = make_contiguous(M, N, K, false, kF32, 1560, 2);
    const uint16_t *raw = whole.wei.as<uint16_t>();
    f16_call_t s;
    s.M = M;
    s.N = N;
    s.K = K / 2;
    s.dst_dt = kF32;
    s.threads = 4;
    direct_t view = make_direct(s, 1561, 1);
    for (int k = 0; k < s.K; ++k)
        for (int n = 0; n < N; ++n)
            view.o.wei[static_cast<size_t>(k) * s.ldb() + n]
                    = raw[static_cast<size_t>(K / 2 + k) * N + n];
    const std::vector<float> want = model_output(s, view.o);
    view.call.wei = raw + static_cast<size_t>(K / 2) * N;
    view.call.ldb = N;
    ASSERT_EQ(nt::run_matmul(view.call), status_t::success);
    EXPECT_TRUE(same_bits(s, view.dst, want));
    ASSERT_EQ(nt::run_matmul(whole.call), status_t::success);
    ASSERT_TRUE(nat::native_weight_packed_in_place(
            whole.wei.get(), whole.wei.bytes));
    for (const int32_t wc : {1, 2}) {
        std::memcpy(view.dst.get(), view.o.dst0.get(), view.o.dst0.bytes);
        view.call.params.weight_cache_type = wc;
        ASSERT_EQ(nt::run_matmul(view.call), status_t::success)
                << "weight cache mode " << wc;
        EXPECT_TRUE(same_bits(s, view.dst, want)) << "weight cache mode " << wc;
    }
    std::memcpy(view.dst.get(), view.o.dst0.get(), view.o.dst0.bytes);
    view.call.params.weight_cache_type = 0;
    EXPECT_EQ(nt::run_matmul(view.call), status_t::failure);
    EXPECT_EQ(std::memcmp(view.dst.get(), view.o.dst0.get(), view.o.dst0.bytes),
            0)
            << "dst written";
    mm::clear_matmul_weight_caches();
}

// The raw weight written back over its in-place image with no clear, as a
// model unloaded and loaded again at the same address, is read as that raw
// weight by the next call: a non-transposed weight, whose image holds the
// raw weight's bytes at both of its ends, and a transposed one, of images of
// 8 KiB and 128 KiB.
TEST(NativeBrgemmF16Route, ARawWeightWrittenBackOverItsImageIsReadAsRaw) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    WeightCacheGuard cache(2);
    uint32_t seed = 1570;
    for (const int K : {64, 1024})
        for (const bool transB : {false, true}) {
            SCOPED_TRACE("K=" + std::to_string(K)
                    + (transB ? " transB" : " not transposed"));
            direct_t d = make_contiguous(4, 64, K, transB, kF32, ++seed, 2);
            const std::vector<float> want = model_of(d);
            const std::vector<uint8_t> raw(
                    d.wei.as<uint8_t>(), d.wei.as<uint8_t>() + d.wei.bytes);
            ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
            EXPECT_TRUE(same_bits(d.shape, d.dst, want));
            ASSERT_TRUE(nat::native_weight_packed_in_place(
                    d.wei.get(), d.wei.bytes));
            std::memcpy(d.wei.get(), raw.data(), raw.size());
            std::memcpy(d.dst.get(), d.o.dst0.get(), d.o.dst0.bytes);
            ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
            EXPECT_TRUE(same_bits(d.shape, d.dst, want));
            mm::clear_matmul_weight_caches();
        }
}

// The raw bytes of a non-transposed f16 image of N 64 from byte 64 on (its
// first line is the raw weight's own) written back and called as a weight
// of their own leave no part of the record that a reload could not change:
// once the whole raw weight is back, the call on the whole weight gives the
// model's bits, with a clear before the reload or none.  Images of 8 KiB
// and of 128 KiB, whose record keeps its ends and sample lines.
TEST(NativeBrgemmF16Route, AWrittenBackSuffixLeavesNoPartAReloadCannotChange) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    WeightCacheGuard cache(2);
    for (const int K : {64, 1024})
        for (const bool cleared : {false, true}) {
            SCOPED_TRACE("K=" + std::to_string(K)
                    + (cleared ? ", a clear before the reload" : ", no clear"));
            direct_t whole = make_contiguous(4, 64, K, false, kF32, 1590, 2);
            const std::vector<float> want = model_of(whole);
            const std::vector<uint8_t> raw(whole.wei.as<uint8_t>(),
                    whole.wei.as<uint8_t>() + whole.wei.bytes);
            ASSERT_EQ(nt::run_matmul(whole.call), status_t::success);
            ASSERT_TRUE(nat::native_weight_packed_in_place(
                    whole.wei.get(), whole.wei.bytes));
            // A transposed weight of K 32 over the bytes past the first 64.
            const int n = static_cast<int>((raw.size() - 64) / 64);
            direct_t suffix = make_contiguous(4, n, 32, true, kF32, 1591, 2);
            ASSERT_EQ(suffix.wei.bytes, raw.size() - 64);
            std::memcpy(suffix.wei.get(), raw.data() + 64, suffix.wei.bytes);
            std::memcpy(whole.wei.as<uint8_t>() + 64, raw.data() + 64,
                    suffix.wei.bytes);
            const std::vector<float> suffix_want = model_of(suffix);
            suffix.call.wei = whole.wei.as<uint8_t>() + 64;
            ASSERT_EQ(nt::run_matmul(suffix.call), status_t::success);
            EXPECT_TRUE(same_bits(suffix.shape, suffix.dst, suffix_want));
            if (cleared) mm::clear_matmul_weight_caches();
            std::memcpy(whole.wei.get(), raw.data(), raw.size());
            std::memcpy(
                    whole.dst.get(), whole.o.dst0.get(), whole.o.dst0.bytes);
            ASSERT_EQ(nt::run_matmul(whole.call), status_t::success);
            EXPECT_TRUE(same_bits(whole.shape, whole.dst, want));
            mm::clear_matmul_weight_caches();
        }
}

// What matmul_direct requires of the CPU for f16 operands: a call that asks
// for the native BRGEMM, which widens and rounds them with AVX-512 F, runs
// without AVX512-FP16, f16 bias and binary operand included; a call that
// names another route keeps that route's requirement, isa_unsupported on a
// CPU without AVX512-FP16, dst untouched.
TEST(NativeBrgemmF16Route, F16NeedsAvx512Fp16OnlyOffTheNativeBrgemm) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    f16_call_t s;
    s.M = 4;
    s.N = 64;
    s.K = 96;
    s.bias_dt = kF16;
    s.binary = 1;
    s.threads = 4;
    for (const bool per_call : {true, false}) {
        SCOPED_TRACE(per_call ? "lowoha_algo=native_brgemm" : "ALGO 10");
        const nt::algo_pin_t pin(
                per_call ? matmul_algo_t::none : matmul_algo_t::native_brgemm);
        direct_t d = make_direct(s, 1700);
        if (per_call) d.call.params.lowoha_algo = matmul_algo_t::native_brgemm;
        ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
        EXPECT_TRUE(same_bits(s, d.dst, model_of(d)));
    }
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    for (const matmul_algo_t other :
            {matmul_algo_t::reference, matmul_algo_t::aocl_dlp_blocked}) {
        direct_t d = make_direct(s, 1710);
        d.call.params.lowoha_algo = other;
        const status_t st = nt::run_matmul(d.call);
        if (has_avx512_fp16()) continue;
        EXPECT_EQ(st, status_t::isa_unsupported);
        EXPECT_EQ(std::memcmp(d.dst.get(), d.o.dst0.get(), d.o.dst0.bytes), 0);
    }
    mm::clear_matmul_weight_caches();
}

/// d's call with its f16 source quantized to s8 in the call, over w, an s8
/// weight with a scale per column or an s4 weight with {2, N} scales.
void quantize_source(direct_t &d, data_type_t wei, const buffer_t &w,
        std::vector<float> &wscale) {
    const int N = d.shape.N;
    const int groups = wei == data_type_t::s4 ? 2 : 1;
    wscale.assign(static_cast<size_t>(groups) * N, 0.01f);
    d.call.wei = w.get();
    d.call.params.dtypes.wei = wei;
    d.call.params.dtypes.compute = data_type_t::s8;
    d.call.params.dynamic_quant = true;
    d.call.params.quant_params.wei_scale.buff = wscale.data();
    d.call.params.quant_params.wei_scale.dt = kF32;
    d.call.params.quant_params.wei_scale.dims = {groups, N};
}

/// The status of d's call through matmul_direct, d's dst restored first.
status_t run_column_major(direct_t &d) {
    std::memcpy(d.dst.get(), d.o.dst0.get(), d.o.dst0.bytes);
    mm::matmul_batch_params_t batch;
    const f16_call_t &s = d.shape;
    return mm::matmul_direct('c', false, false, s.M, s.N, s.K, 1.0f,
            d.o.src.data(), s.M, d.wei.get(), s.K, nullptr, 0.0f, d.dst.get(),
            s.M, d.call.is_weights_const, batch, d.call.params);
}

/// The status native_brgemm_refusal gives d's call made column-major.
status_t column_major_refusal(const direct_t &d) {
    mm::matmul_batch_params_t batch;
    const f16_call_t &s = d.shape;
    return mm::native_brgemm_refusal('c', false, false, s.M, s.N, s.K, 1.0f,
            d.o.src.data(), s.M, d.wei.get(), s.K, nullptr, 0.0f, d.dst.get(),
            s.M, d.call.is_weights_const, batch, d.call.params);
}

bool dst_untouched(const direct_t &d) {
    return std::memcmp(d.dst.get(), d.o.dst0.get(), d.o.dst0.bytes) == 0;
}

// A new weight of 64 bytes written with no clear where a non-transposed
// 64 x 64 f16 image starts, which holds neither line a call the image serves
// checks (the image's first and last lines are the raw weight's own), is
// checked at the image's bytes it holds, from the thread whose memo holds
// the image and from another: the column-major call of K 16, N 2 and ldb 16
// on it is not refused as one on the image (the native BRGEMM hands it
// back), and the transposed row-major call on the same bytes gives the
// model's bits.
TEST(NativeBrgemmF16Route, AShortWeightAtAnImagesStartIsCheckedAtItsBytes) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    WeightCacheGuard cache(2);
    for (const bool same_thread : {true, false}) {
        SCOPED_TRACE(same_thread ? "the thread that wrote the image"
                                 : "another thread");
        direct_t whole = make_contiguous(4, 64, 64, false, kF32, 1580, 2);
        ASSERT_EQ(nt::run_matmul(whole.call), status_t::success);
        ASSERT_TRUE(nat::native_weight_packed_in_place(
                whole.wei.get(), whole.wei.bytes));
        direct_t fresh = make_contiguous(4, 2, 16, true, kF32, 1581, 2);
        const std::vector<float> want = model_of(fresh);
        std::memcpy(whole.wei.get(), fresh.wei.get(), fresh.wei.bytes);
        f16_call_t s;
        s.M = 4;
        s.N = 2;
        s.K = 16;
        s.dst_dt = kF32;
        direct_t column = make_direct(s, 1582, 2);
        column.wei = whole.wei;
        status_t st = status_t::success;
        if (same_thread) {
            st = run_column_major(column);
        } else {
            std::thread other([&] { st = run_column_major(column); });
            other.join();
        }
        EXPECT_NE(st, status_t::failure) << "refused as a call on the image";
        fresh.call.wei = whole.wei.get();
        ASSERT_EQ(nt::run_matmul(fresh.call), status_t::success);
        EXPECT_TRUE(same_bits(fresh.shape, fresh.dst, want));
        mm::clear_matmul_weight_caches();
    }
}

// A call that asks for the native BRGEMM takes an f16 bias, binary operand
// or dst of a bf16, s8 or f32 weight on a CPU without AVX512-FP16 too:
// under ZENDNNL_MATMUL_ALGO=10 and with lowoha_algo native_brgemm, bf16 x
// bf16, u8 x s8 and f32 x f32 calls with an f16 bias or an f16 binary_add
// operand give the bits of the same call with the operand widened to f32,
// and u8 x s8 and bf16 x bf16 calls into an f16 dst those bits rounded to
// f16.  So do their calls of one row on a constant weight at weight cache
// modes 1 and 2, on the tiles as every call.
TEST(NativeBrgemmF16Route, F16OperandsOfOtherCallsNeedNoAvx512Fp16) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const int rows = 5, N = 64, K = 96;
    const std::vector<uint16_t> bias16 = random_f16(N, 2200);
    const std::vector<uint16_t> bin16
            = random_f16(static_cast<size_t>(rows) * N, 2201);
    std::vector<float> bias32(N), bin32(bin16.size());
    for (size_t i = 0; i < bias32.size(); ++i)
        bias32[i] = f32_of(bias16[i]);
    for (size_t i = 0; i < bin32.size(); ++i)
        bin32[i] = f32_of(bin16[i]);
    const float src_scale = 0.02f;
    const std::vector<float> wei_scales(N, 0.01f);
    enum class what_t { bias, binary, dst };
    struct case_t {
        data_type_t wei;
        what_t what;
        bool per_call;
        int M;
        int32_t wc;
    };
    // k's call of the source rows a on the weight w: as f16 the call under
    // test, on a constant weight; else the same call with the operand
    // widened (and an f32 dst), on a weight that is not constant.
    const auto call = [&](const case_t &k, const buffer_t &a, const buffer_t &w,
                              bool as_f16, buffer_t &dst) {
        const bool int8 = k.wei == data_type_t::s8;
        nt::call_t c;
        c.M = k.M;
        c.N = N;
        c.K = K;
        c.src = a.get();
        c.wei = w.get();
        c.is_weights_const = as_f16;
        c.params.dtypes.src = int8 ? data_type_t::u8 : k.wei;
        c.params.dtypes.wei = k.wei;
        c.params.dtypes.dst = as_f16 && k.what == what_t::dst ? kF16 : kF32;
        c.params.num_threads = 4;
        c.params.weight_cache_type = k.wc;
        if (int8) {
            auto &q = c.params.quant_params;
            q.src_scale.buff = &src_scale;
            q.src_scale.dt = kF32;
            q.src_scale.dims = {1, 1};
            q.wei_scale.buff = wei_scales.data();
            q.wei_scale.dt = kF32;
            q.wei_scale.dims = {1, N};
        }
        if (k.what == what_t::bias) {
            c.params.dtypes.bias = as_f16 ? kF16 : kF32;
            c.bias = as_f16 ? static_cast<const void *>(bias16.data())
                            : bias32.data();
        }
        if (k.what == what_t::binary) {
            mm::matmul_post_op po;
            po.po_type = post_op_type_t::binary_add;
            po.buff = as_f16
                    ? static_cast<void *>(const_cast<uint16_t *>(bin16.data()))
                    : static_cast<void *>(bin32.data());
            po.dtype = as_f16 ? kF16 : kF32;
            po.dims = {k.M, N};
            po.leading_dim = N;
            c.params.postop_.push_back(po);
        }
        if (k.per_call) c.params.lowoha_algo = matmul_algo_t::native_brgemm;
        dst = nt::make_buffer(static_cast<size_t>(k.M) * N
                * (c.params.dtypes.dst == kF16 ? 2 : 4));
        c.dst = dst.get();
        return nt::run_matmul(c);
    };
    int ran = 0;
    for (const data_type_t wei : {kBf16, data_type_t::s8, kF32}) {
        const bool int8 = wei == data_type_t::s8;
        if (wei == kBf16 && !nat::detect_uarch().avx512bf16) continue;
        if (int8 && !nat::detect_uarch().avx512vnni) continue;
        const buffer_t a = nt::random_buffer(int8 ? data_type_t::u8 : wei,
                static_cast<size_t>(rows) * K, 2210, int8 ? 0 : -1,
                int8 ? 9 : 1);
        const buffer_t raw = nt::random_buffer(wei, static_cast<size_t>(K) * N,
                2211, int8 ? -9 : -1, int8 ? 9 : 1);
        for (const what_t what : {what_t::bias, what_t::binary, what_t::dst})
            for (const bool per_call : {false, true})
                for (const int M : {rows, 1})
                    for (const int32_t wc : {1, 2}) {
                        // The checks take no f16 dst of an f32 source.
                        if (what == what_t::dst && wei == kF32) continue;
                        const case_t k {wei, what, per_call, M, wc};
                        SCOPED_TRACE("wei "
                                + std::to_string(static_cast<int>(wei))
                                + " what "
                                + std::to_string(static_cast<int>(what))
                                + (per_call ? " per call" : " ALGO 10") + " M "
                                + std::to_string(M) + " weight cache "
                                + std::to_string(wc));
                        const nt::algo_pin_t pin(per_call
                                        ? matmul_algo_t::none
                                        : matmul_algo_t::native_brgemm);
                        WeightCacheGuard cache(wc);
                        // A weight of its own: a call at mode 2 may rewrite
                        // it in place.
                        const buffer_t w = nt::kept_until_exit(
                                nt::make_buffer(raw.bytes));
                        std::memcpy(w.get(), raw.get(), raw.bytes);
                        buffer_t widened, got;
                        ASSERT_EQ(call(k, a, w, false, widened),
                                status_t::success);
                        ASSERT_EQ(call(k, a, w, true, got), status_t::success);
                        int bad = 0;
                        for (int i = 0; i < M * N; ++i) {
                            const float v = widened.as<float>()[i];
                            bad += what == what_t::dst
                                    ? got.as<uint16_t>()[i] != f16_of(v)
                                    : std::memcmp(&got.as<float>()[i], &v, 4)
                                            != 0;
                        }
                        EXPECT_EQ(bad, 0);
                        EXPECT_NE(nt::native_layouts(w.get()), 0u);
                        ++ran;
                    }
    }
    EXPECT_GT(ran, 0);
    mm::clear_matmul_weight_caches();
}

// The f16 calls the native BRGEMM does not serve, each with the status the
// tests of the native suites take from native_brgemm_refusal, and dst and
// the weight untouched.  With lowoha_algo native_brgemm: an f16 source
// quantized to s8 for an s8 weight and an f16 weight with a bf16 source
// (unimplemented, with their reasons); under ZENDNNL_MATMUL_ALGO=10 too: an
// f16 call into an s8 dst and a column-major call on a constant weight,
// whose errors the hand-back keeps (unimplemented), and a batched call that
// names the native BRGEMM (unimplemented).  matmul_direct's checks refuse
// before any route an f16 source quantized to s8 for an s4 weight, an f16
// call with a weight zero point and an f16 call into a bf16 dst (failure).
TEST(NativeBrgemmF16Route, OutOfScopeCallsAreRefused) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    f16_call_t s;
    s.M = 4;
    s.N = 64;
    s.K = 64;
    s.threads = 4;
    {
        // DA8W8 with an f16 source.
        direct_t d = make_direct(s, 1800);
        const buffer_t w8 = nt::random_buffer(data_type_t::s8,
                static_cast<size_t>(s.K) * s.ldb(), 3, -127, 127);
        std::vector<float> wscale;
        quantize_source(d, data_type_t::s8, w8, wscale);
        d.call.params.lowoha_algo = matmul_algo_t::native_brgemm;
        std::string why;
        EXPECT_EQ(nt::refusal_of(d.call, &why), status_t::unimplemented);
        EXPECT_EQ(why, "s8 weight with a source other than u8 / s8");
        EXPECT_TRUE(nt::refused_with(d.call, status_t::unimplemented));
        EXPECT_TRUE(dst_untouched(d));
        EXPECT_EQ(nt::native_layouts(w8.get()), 0u);
    }
    {
        // bf16 source with an f16 weight.
        f16_call_t q = s;
        q.dst_dt = kF32;
        direct_t d = make_direct(q, 1810);
        const buffer_t src = nt::random_buffer(
                kBf16, static_cast<size_t>(q.M) * q.lda(), 5, -1, 1);
        d.call.src = src.get();
        d.call.params.dtypes.src = kBf16;
        d.call.params.lowoha_algo = matmul_algo_t::native_brgemm;
        std::string why;
        EXPECT_EQ(nt::refusal_of(d.call, &why), status_t::unimplemented);
        EXPECT_EQ(why, "f16 weight with a non-f16 source");
        EXPECT_TRUE(nt::refused_with(d.call, status_t::unimplemented));
        EXPECT_TRUE(dst_untouched(d));
    }
    for (const bool per_call : {false, true}) {
        SCOPED_TRACE(per_call ? "lowoha_algo=native_brgemm" : "ALGO 10");
        {
            // An s8 dst.
            f16_call_t q = s;
            q.dst_dt = kF32;
            direct_t d = make_direct(q, 1820);
            std::vector<int8_t> dst(static_cast<size_t>(q.M) * q.N, 7);
            d.call.dst = dst.data();
            d.call.params.dtypes.dst = data_type_t::s8;
            if (per_call)
                d.call.params.lowoha_algo = matmul_algo_t::native_brgemm;
            std::string why;
            EXPECT_EQ(nt::refusal_of(d.call, &why), status_t::unimplemented);
            EXPECT_EQ(why, "an f16 call into a dst other than f16 or f32");
            EXPECT_TRUE(nt::refused_with(d.call, status_t::unimplemented));
            EXPECT_TRUE(std::all_of(
                    dst.begin(), dst.end(), [](int8_t v) { return v == 7; }));
        }
        {
            // Column-major on a constant weight.
            direct_t d = make_direct(s, 1830);
            if (per_call)
                d.call.params.lowoha_algo = matmul_algo_t::native_brgemm;
            const status_t refused = column_major_refusal(d);
            EXPECT_EQ(refused, status_t::unimplemented);
            EXPECT_EQ(run_column_major(d), refused);
            EXPECT_TRUE(dst_untouched(d));
            EXPECT_EQ(nt::native_layouts(d.wei.get()), 0u);
        }
        {
            // Two batches that name the native BRGEMM.
            direct_t d = make_direct(s, 1840);
            std::vector<uint16_t> src(2 * d.o.src.size());
            std::copy(d.o.src.begin(), d.o.src.end(), src.begin());
            std::copy(d.o.src.begin(), d.o.src.end(),
                    src.begin() + d.o.src.size());
            std::vector<uint16_t> dst(
                    2 * static_cast<size_t>(s.M) * s.ldc(), kUnwrittenF16);
            mm::matmul_batch_params_t batch;
            batch.Batch_A = 2;
            batch.Batch_B = 1;
            mm::matmul_params p = d.call.params;
            p.lowoha_algo = matmul_algo_t::native_brgemm;
            const status_t refused = mm::native_brgemm_refusal('r', false,
                    false, s.M, s.N, s.K, 1.0f, src.data(), s.lda(),
                    d.wei.get(), s.ldb(), nullptr, 0.0f, dst.data(), s.ldc(),
                    true, batch, p);
            EXPECT_EQ(refused, status_t::unimplemented);
            const status_t st = mm::matmul_direct('r', false, false, s.M, s.N,
                    s.K, 1.0f, src.data(), s.lda(), d.wei.get(), s.ldb(),
                    nullptr, 0.0f, dst.data(), s.ldc(), true, batch, p);
            EXPECT_EQ(st, refused);
            EXPECT_TRUE(std::all_of(dst.begin(), dst.end(),
                    [](uint16_t v) { return v == kUnwrittenF16; }));
        }
        {
            // W4A8 with an f16 source: the checks take W4A8 only from an s8
            // or a bf16 / f32 source.
            direct_t d = make_direct(s, 1850);
            const buffer_t w4 = nt::random_buffer(data_type_t::s8,
                    static_cast<size_t>(s.K) * s.N / 2, 4, -127, 127);
            std::vector<float> wscale;
            quantize_source(d, data_type_t::s4, w4, wscale);
            d.call.ldb = s.N;
            if (per_call)
                d.call.params.lowoha_algo = matmul_algo_t::native_brgemm;
            EXPECT_EQ(nt::refusal_of(d.call), status_t::success);
            EXPECT_EQ(nt::run_matmul(d.call), status_t::failure);
            EXPECT_TRUE(dst_untouched(d));
            EXPECT_EQ(nt::native_layouts(w4.get()), 0u);
        }
        {
            // A weight zero point.
            direct_t d = make_direct(s, 1860);
            const int8_t zp = 1;
            d.call.params.quant_params.wei_zp.buff = &zp;
            d.call.params.quant_params.wei_zp.dt = data_type_t::s8;
            d.call.params.quant_params.wei_zp.dims = {1, 1};
            if (per_call)
                d.call.params.lowoha_algo = matmul_algo_t::native_brgemm;
            EXPECT_EQ(nt::refusal_of(d.call), status_t::success);
            EXPECT_EQ(nt::run_matmul(d.call), status_t::failure);
            EXPECT_TRUE(dst_untouched(d));
        }
        {
            // A bf16 dst.
            f16_call_t q = s;
            q.dst_dt = kF32;
            direct_t d = make_direct(q, 1870);
            std::vector<uint16_t> dst(static_cast<size_t>(q.M) * q.N, 0x3F80);
            d.call.dst = dst.data();
            d.call.params.dtypes.dst = kBf16;
            if (per_call)
                d.call.params.lowoha_algo = matmul_algo_t::native_brgemm;
            EXPECT_EQ(nt::refusal_of(d.call), status_t::success);
            EXPECT_EQ(nt::run_matmul(d.call), status_t::failure);
            EXPECT_TRUE(std::all_of(dst.begin(), dst.end(),
                    [](uint16_t v) { return v == 0x3F80; }));
        }
    }
    mm::clear_matmul_weight_caches();
}

// Under ZENDNNL_MATMUL_ALGO=10 the native BRGEMM hands the f16 calls it does
// not implement, and that the default route computes correctly, back to the
// default route (native_brgemm_refusal gives success): a column-major call
// on a weight that changes returns the status and dst of the same call with
// the algorithm unset, on a CPU without AVX512-FP16 isa_unsupported from
// the default route's check, before any other route reads the call, dst and
// the weight untouched.  The forms the default route does not compute keep
// the native error (unimplemented) with dst untouched: an f16 source
// quantized to s8 for an s8 weight (it quantizes bf16 and f32 sources
// only), a bf16 source with an f16 weight (it has no kernel for the two),
// and in a build without AOCL-DLP the column-major call.
TEST(NativeBrgemmF16Route, HandedBackCallsMeetTheDefaultRoutesIsa) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    f16_call_t s;
    s.M = 4;
    s.N = 64;
    s.K = 64;
    s.dst_dt = kF32;
    s.threads = 4;
    const buffer_t w8 = nt::random_buffer(
            data_type_t::s8, static_cast<size_t>(s.K) * s.ldb(), 3, -127, 127);
    const buffer_t bf16_src = nt::random_buffer(
            kBf16, static_cast<size_t>(s.M) * s.lda(), 5, -1, 1);
    std::vector<float> wscale;
    enum class form_t { quantized_source, bf16_source, column_major };
    for (const form_t form : {form_t::quantized_source, form_t::bf16_source,
                 form_t::column_major}) {
        SCOPED_TRACE("form " + std::to_string(static_cast<int>(form)));
        direct_t d = make_direct(s, 1900 + static_cast<int>(form));
        if (form == form_t::quantized_source)
            quantize_source(d, data_type_t::s8, w8, wscale);
        if (form == form_t::bf16_source) {
            d.call.src = bf16_src.get();
            d.call.params.dtypes.src = kBf16;
        }
        const bool column_major = form == form_t::column_major;
        if (column_major) d.call.is_weights_const = false;
        const auto run = [&]() {
            if (column_major) return run_column_major(d);
            std::memcpy(d.dst.get(), d.o.dst0.get(), d.o.dst0.bytes);
            return nt::run_matmul(d.call);
        };
        status_t unset = status_t::success;
        std::vector<uint8_t> unset_dst;
        {
            const nt::algo_pin_t none(matmul_algo_t::none);
            unset = run();
            unset_dst.assign(
                    d.dst.as<uint8_t>(), d.dst.as<uint8_t>() + d.dst.bytes);
        }
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        const bool kept = !column_major || !ZENDNNL_DEPENDS_AOCLDLP;
        EXPECT_EQ(
                column_major ? column_major_refusal(d) : nt::refusal_of(d.call),
                kept ? status_t::unimplemented : status_t::success);
        if (kept) {
            EXPECT_EQ(run(), status_t::unimplemented);
            EXPECT_TRUE(dst_untouched(d));
        } else {
            EXPECT_EQ(run(), unset);
            EXPECT_EQ(
                    std::memcmp(d.dst.get(), unset_dst.data(), d.dst.bytes), 0);
            if (!has_avx512_fp16()) {
                EXPECT_EQ(unset, status_t::isa_unsupported);
                EXPECT_TRUE(dst_untouched(d));
            }
        }
        EXPECT_EQ(nt::native_layouts(d.call.wei), 0u);
    }
    mm::clear_matmul_weight_caches();
}

// A column-major f16 call, not transposed, of K 32, N 96 and ldb 32 on a
// weight that changes reads N columns of K elements, 6,144 bytes, and its
// hold spans them all.  While the hold is kept, a first native call at
// weight cache mode 2 on a 16 x 64 f16 weight 4,096 bytes into the buffer
// waits to rewrite it in place, and every byte of the span reads as it
// was.  Once that weight is an image, the hold finds it and fails, and the
// call under ZENDNNL_MATMUL_ALGO=10, which reads into the image, fails with
// dst untouched: no route reads it, in every build and on every CPU.
TEST(NativeBrgemmF16Route, AHandedBackColumnMajorCallHoldsAllItsBytes) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    WeightCacheGuard cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int M = 4, K = 32, N = 96, ldb = 32, VK = 16, VN = 64;
    const size_t span = static_cast<size_t>(N) * ldb * 2, offset = 4096;
    ASSERT_EQ(offset + static_cast<size_t>(VK) * VN * 2, span);
    const std::vector<uint16_t> raw = random_f16(span / 2, 2200);
    const std::vector<uint16_t> view_src
            = random_f16(static_cast<size_t>(M) * VK, 2201);
    const std::vector<uint16_t> src
            = random_f16(static_cast<size_t>(M) * K, 2202);
    const auto view_call = [&](const uint8_t *w, float *dst) {
        mm::matmul_batch_params_t batch;
        mm::matmul_params p;
        p.dtypes.src = p.dtypes.wei = kF16;
        p.dtypes.dst = kF32;
        p.weight_cache_type = 2;
        p.num_threads = 1;
        return mm::matmul_direct('r', false, false, M, VN, VK, 1.0f,
                view_src.data(), VK, w, VN, nullptr, 0.0f, dst, VN, true, batch,
                p);
    };
    mm::matmul_params cm;
    cm.dtypes.src = cm.dtypes.wei = kF16;
    cm.dtypes.dst = kF32;
    cm.weight_cache_type = 2;
    {
        buffer_t shared = nt::kept_until_exit(nt::make_buffer(span));
        std::memcpy(shared.get(), raw.data(), span);
        auto hold = std::make_unique<nat::native_weight_raw_hold_t>();
        ASSERT_EQ(nat::native_hold_handed_back_weight(
                          *hold, 'c', shared.get(), K, N, ldb, false, cm),
                status_t::success);
        std::vector<float> dst(static_cast<size_t>(M) * VN);
        std::atomic<bool> done {false};
        status_t st = status_t::failure;
        std::thread native([&] {
            st = view_call(shared.as<uint8_t>() + offset, dst.data());
            done.store(true);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::vector<uint8_t> read(span);
        std::memcpy(read.data(), shared.get(), span);
        const bool waited = !done.load();
        hold.reset();
        native.join();
        EXPECT_TRUE(waited) << "the native call did not wait for the hold";
        EXPECT_EQ(std::memcmp(read.data(), raw.data(), span), 0)
                << "an in-place rewrite wrote bytes the hold holds";
        EXPECT_EQ(st, status_t::success);
        EXPECT_TRUE(nat::native_weight_packed_in_place(
                shared.as<uint8_t>() + offset, span - offset));
        mm::clear_matmul_weight_caches();
    }
    buffer_t shared = nt::kept_until_exit(nt::make_buffer(span));
    std::memcpy(shared.get(), raw.data(), span);
    std::vector<float> view_dst(static_cast<size_t>(M) * VN);
    ASSERT_EQ(view_call(shared.as<uint8_t>() + offset, view_dst.data()),
            status_t::success);
    ASSERT_TRUE(nat::native_weight_packed_in_place(
            shared.as<uint8_t>() + offset, span - offset));
    {
        nat::native_weight_raw_hold_t hold;
        EXPECT_EQ(nat::native_hold_handed_back_weight(
                          hold, 'c', shared.get(), K, N, ldb, false, cm),
                status_t::failure);
    }
    const std::vector<float> prior(static_cast<size_t>(M) * N, -7.0f);
    std::vector<float> dst = prior;
    mm::matmul_batch_params_t batch;
    const status_t st = mm::matmul_direct('c', false, false, M, N, K, 1.0f,
            src.data(), M, shared.get(), ldb, nullptr, 0.0f, dst.data(), M,
            false, batch, cm);
    EXPECT_EQ(st, status_t::failure);
    EXPECT_EQ(std::memcmp(dst.data(), prior.data(), dst.size() * sizeof(float)),
            0)
            << "dst written";
    mm::clear_matmul_weight_caches();
}

// The buffer of a call's widened source outlives the call only within the
// process-wide bound, and clear_matmul_native_weight_caches frees it: calls
// one after the other, on two threads, keep one buffer; a call whose
// widened source is larger than the bound keeps nothing; the clear leaves
// nothing kept, also of a buffer a call held while it ran.
TEST(NativeBrgemmF16Route, WidenedSourcesAreBoundedAndReleased) {
    SKIP_WITHOUT_NATIVE_ISA();
    mm::clear_matmul_weight_caches();
    EXPECT_EQ(nat::native_f16_kept_source_bytes(), 0u);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    f16_call_t s;
    s.M = 64;
    s.N = 64;
    s.K = 1024;
    s.threads = 4;
    direct_t d = make_direct(s, 2100);
    const std::vector<float> want = model_of(d);
    ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
    EXPECT_TRUE(same_bits(s, d.dst, want));
    const size_t one = nat::native_f16_kept_source_bytes();
    // The F16_PH kernels read the call's source rows: nothing is widened.
    if (f16_ph_calls()) {
        EXPECT_EQ(one, 0u);
    } else {
        EXPECT_GE(one, static_cast<size_t>(s.M) * (s.K + 16) * sizeof(float));
    }
    std::thread other([&] {
        std::memcpy(d.dst.get(), d.o.dst0.get(), d.o.dst0.bytes);
        EXPECT_EQ(nt::run_matmul(d.call), status_t::success);
    });
    other.join();
    EXPECT_TRUE(same_bits(s, d.dst, want));
    EXPECT_EQ(nat::native_f16_kept_source_bytes(), one);
    mm::clear_matmul_native_weight_caches();
    EXPECT_EQ(nat::native_f16_kept_source_bytes(), 0u);

    // 8200 x 16400 floats: past the bound of 512 MiB.  The source is zero
    // pages the call only reads.
    const int M = 8200, N = 32, K = 16384;
    std::unique_ptr<uint16_t, decltype(&std::free)> src(
            static_cast<uint16_t *>(
                    std::calloc(static_cast<size_t>(M) * K, sizeof(uint16_t))),
            &std::free);
    ASSERT_NE(src, nullptr);
    const std::vector<uint16_t> wei
            = random_f16(static_cast<size_t>(K) * N, 2110);
    std::vector<uint16_t> dst(static_cast<size_t>(M) * N, kUnwrittenF16);
    nt::call_t big;
    big.M = M;
    big.N = N;
    big.K = K;
    big.src = src.get();
    big.wei = wei.data();
    big.dst = dst.data();
    big.params.dtypes.src = big.params.dtypes.wei = big.params.dtypes.dst
            = kF16;
    big.params.num_threads = 4;
    ASSERT_EQ(nt::run_matmul(big), status_t::success);
    EXPECT_EQ(dst[0], 0u);
    EXPECT_EQ(nat::native_f16_kept_source_bytes(), 0u);
    ASSERT_EQ(nt::run_matmul(d.call), status_t::success);
    EXPECT_EQ(nat::native_f16_kept_source_bytes(), one);
    mm::clear_matmul_weight_caches();
    EXPECT_EQ(nat::native_f16_kept_source_bytes(), 0u);
}

// On the widening kernels a call's M widened rows, K floats apart or K + 16
// at a multiple of 1024, go into a kept buffer of exactly those rows
// rounded up to 64 bytes: calls of M 1, 2 and 4, also of transposed sources,
// each with the model's bits and the kept bytes after it.  In a child with
// ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32, which keeps f16 calls on the widening
// kernels on every CPU.
TEST(NativeBrgemmF16Route, AKeptBufferHoldsTheWidenedRows) {
    SKIP_WITHOUT_NATIVE_ISA();
    nt::expect_in_child({{"ZENDNNL_MATMUL_NATIVE_F16_ACCUM", "f32"}}, [] {
        if (f16_ph_calls()) {
            std::fprintf(stderr, "f16 calls are not on the widening kernels\n");
            return 1;
        }
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        struct rows_t {
            int M, K;
            bool transA;
            size_t kept;
        };
        const rows_t calls[] = {{1, 256, false, 256 * 4},
                {1, 4112, false, 4112 * 4}, {2, 2056, false, 2 * 2056 * 4},
                {4, 1028, true, 4 * 1028 * 4}, {4, 1024, false, 4 * 1040 * 4},
                {2, 2048, false, 2 * 2064 * 4}, {4, 1029, false, 16512},
                {4, 1024, true, 4 * 1040 * 4}};
        int bad = 0;
        uint32_t seed = 2200;
        for (const rows_t &r : calls) {
            nat::release_native_f16_sources();
            f16_call_t s;
            s.M = r.M;
            s.N = 64;
            s.K = r.K;
            s.transA = r.transA;
            direct_t d = make_direct(s, seed++);
            const std::vector<float> want = model_of(d);
            const status_t st = nt::run_matmul(d.call);
            const size_t kept = nat::native_f16_kept_source_bytes();
            const bool bits = same_bits(s, d.dst, want);
            if (st != status_t::success || !bits || kept != r.kept) {
                std::fprintf(stderr,
                        "%s: status %d, %s bits, kept %zu (expected %zu)\n",
                        str(s).c_str(), static_cast<int>(st),
                        bits ? "the model's" : "not the model's", kept, r.kept);
                ++bad;
            }
        }
        nat::release_native_f16_sources();
        return bad;
    }, "ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32");
}

// An f16 call on the widening kernels takes at most 2 KiB more of the
// calling thread's stack than a bf16 call of its size, and both run on a
// thread of a 64 KiB stack: warm M = 1 calls on one thread, each on a
// painted stack of its own above a guard page.  In a child with
// ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32, which keeps f16 calls on the widening
// kernels on every CPU.  Only in optimized builds without a sanitizer, whose
// frames the bound is for (Clang names its sanitizers in __has_feature only).
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) \
        || __has_feature(memory_sanitizer)
#define NATIVE_F16_STACK_TEST_SANITIZED 1
#endif
#endif
TEST(NativeBrgemmF16Route, AWidenedCallTakesTheStackOfOtherCalls) {
#if !defined(__linux__) || !defined(__OPTIMIZE__) \
        || defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__) \
        || defined(NATIVE_F16_STACK_TEST_SANITIZED)
    GTEST_SKIP() << "needs an optimized Linux build without a sanitizer";
#else
    SKIP_WITHOUT_NATIVE_ISA();
    if (!nat::detect_uarch().avx512bf16) GTEST_SKIP() << "needs AVX512-BF16";
    // The bytes of a stack of bytes that fn wrote on a new thread (with the
    // thread's storage at its top), or 0 when no thread ran.
    const auto stack_bytes_used
            = [](size_t bytes, const std::function<void()> &fn) {
        const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        void *map = mmap(nullptr, page + bytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (map == MAP_FAILED) return size_t {0};
        constexpr uint8_t kPaint = 0xa5;
        uint8_t *const low = static_cast<uint8_t *>(map) + page;
        std::memset(low, kPaint, bytes);
        size_t used = 0;
        pthread_attr_t attr;
        if (mprotect(map, page, PROT_NONE) == 0
                && pthread_attr_init(&attr) == 0) {
            const auto run = [](void *f) -> void * {
                (*static_cast<const std::function<void()> *>(f))();
                return nullptr;
            };
            pthread_t thread;
            if (pthread_attr_setstack(&attr, low, bytes) == 0
                    && pthread_create(&thread, &attr, run,
                               const_cast<std::function<void()> *>(&fn))
                            == 0) {
                pthread_join(thread, nullptr);
                size_t clean = 0;
                while (clean < bytes && low[clean] == kPaint)
                    ++clean;
                used = bytes - clean;
            }
            pthread_attr_destroy(&attr);
        }
        munmap(map, page + bytes);
        return used;
    };
    nt::expect_in_child({{"ZENDNNL_MATMUL_NATIVE_F16_ACCUM", "f32"}}, [&] {
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        f16_call_t s;
        s.M = 1;
        s.N = 64;
        s.K = 256;
        direct_t f16 = make_direct(s, 2300);
        const std::vector<float> want = model_of(f16);
        nt::call_t bf16;
        bf16.M = s.M;
        bf16.N = s.N;
        bf16.K = s.K;
        const buffer_t src = nt::random_buffer(kBf16, s.K, 2301, -1, 1);
        const buffer_t wei = nt::kept_until_exit(nt::random_buffer(
                kBf16, static_cast<size_t>(s.K) * s.N, 2302, -1, 1));
        const buffer_t dst = nt::make_buffer(static_cast<size_t>(s.N) * 2);
        bf16.src = src.get();
        bf16.wei = wei.get();
        bf16.dst = dst.get();
        bf16.params.dtypes.src = bf16.params.dtypes.wei = bf16.params.dtypes.dst
                = kBf16;
        bf16.params.num_threads = 1;
        if (nt::run_matmul(f16.call) != status_t::success
                || nt::run_matmul(bf16) != status_t::success) {
            std::fprintf(stderr, "the calls fail on the calling thread\n");
            return 1;
        }
        const size_t stack = 64 * 1024;
        status_t st_f16 = status_t::failure, st_bf16 = status_t::failure;
        const size_t f16_bytes = stack_bytes_used(
                stack, [&] { st_f16 = nt::run_matmul(f16.call); });
        const size_t bf16_bytes = stack_bytes_used(
                stack, [&] { st_bf16 = nt::run_matmul(bf16); });
        const size_t most = bf16_bytes + 2048;
        if (st_f16 != status_t::success || st_bf16 != status_t::success
                || !same_bits(s, f16.dst, want) || bf16_bytes == 0
                || f16_bytes > most) {
            std::fprintf(stderr,
                    "f16: status %d, %zu bytes of stack; bf16: status %d, "
                    "%zu bytes (f16 at most %zu)\n",
                    static_cast<int>(st_f16), f16_bytes,
                    static_cast<int>(st_bf16), bf16_bytes, most);
            return 1;
        }
        return 0;
    }, "ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32");
#endif
}

// A buffer goes back to the pool without allocating: into a free slot, in
// place of the smallest kept buffer when that is smaller, else it is freed,
// as is a buffer a lease held across a release.  The kept bytes are those
// of the buffers in the slots.
TEST(NativeBrgemmF16Route, ABufferThePoolHasNoSlotForIsFreed) {
    nat::release_native_f16_sources();
    ASSERT_EQ(nat::native_f16_kept_source_bytes(), 0u);
    const int slots = nat::kNativeF16KeptSourceSlots;
    const auto bytes_of = [](int i) { return static_cast<size_t>(i + 1) * 64; };
    size_t largest = 0;
    for (int i = 1; i <= slots; ++i)
        largest += bytes_of(i);
    for (const int last : {0, slots}) {
        SCOPED_TRACE(last == 0 ? "the smallest goes back last"
                               : "the largest goes back last");
        std::vector<std::unique_ptr<nat::native_f16_source_lease_t>> leases;
        for (int i = 0; i <= slots; ++i) {
            leases.push_back(
                    std::make_unique<nat::native_f16_source_lease_t>());
            ASSERT_NE(leases.back()->take(bytes_of(i)), nullptr);
        }
        for (int i = 0; i <= slots; ++i)
            if (i != last) leases[i].reset();
        EXPECT_EQ(nat::native_f16_kept_source_bytes(),
                largest + bytes_of(0) - bytes_of(last));
        leases[last].reset();
        EXPECT_EQ(nat::native_f16_kept_source_bytes(), largest);
        nat::release_native_f16_sources();
        EXPECT_EQ(nat::native_f16_kept_source_bytes(), 0u);
    }
    {
        nat::native_f16_source_lease_t held;
        ASSERT_NE(held.take(bytes_of(0)), nullptr);
        nat::release_native_f16_sources();
    }
    EXPECT_EQ(nat::native_f16_kept_source_bytes(), 0u);
}

// A buffer that a call on another thread holds while the kept buffers are
// released is freed when the call gives it back: the call takes it,
// release_native_f16_sources() (or clear_matmul_native_weight_caches(),
// which calls it) runs while the call holds it, and once the call returns
// nothing is kept.  A buffer taken after the release is kept when it goes
// back.
TEST(NativeBrgemmF16Route, ABufferHeldAcrossAReleaseIsFreedWhenItGoesBack) {
    const size_t bytes = 4096;
    for (const bool clear : {false, true}) {
        SCOPED_TRACE(clear ? "clear_matmul_native_weight_caches"
                           : "release_native_f16_sources");
        nat::release_native_f16_sources();
        ASSERT_EQ(nat::native_f16_kept_source_bytes(), 0u);
        std::promise<void> taken, released;
        std::future<void> is_taken = taken.get_future();
        std::future<void> is_released = released.get_future();
        bool held = false;
        std::thread call([&] {
            nat::native_f16_source_lease_t lease;
            held = lease.take(bytes) != nullptr;
            taken.set_value();
            is_released.wait();
        });
        is_taken.wait();
        if (clear)
            mm::clear_matmul_native_weight_caches();
        else
            nat::release_native_f16_sources();
        released.set_value();
        call.join();
        EXPECT_TRUE(held);
        EXPECT_EQ(nat::native_f16_kept_source_bytes(), 0u);
        {
            nat::native_f16_source_lease_t lease;
            ASSERT_NE(lease.take(bytes), nullptr);
        }
        EXPECT_EQ(nat::native_f16_kept_source_bytes(), bytes);
    }
    nat::release_native_f16_sources();
    EXPECT_EQ(nat::native_f16_kept_source_bytes(), 0u);
}

// ZENDNNL_MATMUL_NATIVE_F16_ACCUM, read once: f32 (in any case) keeps f16
// calls on the F16 kernels, with the bits of the widened products added in
// K order; f16, or the variable unset, takes them to the F16_PH kernels on a
// CPU with AVX512-FP16 in a build that has them, with the bits of the
// blocked sums, and is ignored elsewhere; another value keeps the default
// with one warning.  The [MATMUL.NATIVE_BRGEMM] line of a call names its
// kernels, and calls through matmul_direct under ZENDNNL_MATMUL_ALGO=10 run
// on them.
TEST(NativeBrgemmF16Route, TheAccumulationFollowsItsVariable) {
    SKIP_WITHOUT_NATIVE_ISA();
    const bool ph = nat::detect_uarch().avx512fp16
            && nat::select_native_f16_ph_ukernel(1, 2) != nullptr;
    struct setting_t {
        const char *value;
        bool ph;
        bool warns;
    };
    const setting_t settings[]
            = {{"f32", false, false}, {"F32", false, false}, {"f16", ph, false},
                    {nullptr, ph, false}, {"", ph, false}, {"bf16", ph, true}};
    for (const setting_t &s : settings) {
        const std::string what = std::string("ZENDNNL_MATMUL_NATIVE_F16_ACCUM=")
                + (s.value != nullptr ? s.value : "(unset)");
        nt::expect_in_child({{"ZENDNNL_MATMUL_NATIVE_F16_ACCUM", s.value},
                                    {"ZENDNNL_API_LOG_LEVEL", "4"}},
                [&] {
            int bad = 0;
            ::testing::internal::CaptureStdout();
            const bool on_ph = f16_ph_calls();
            f16_call_t c;
            c.M = 9;
            c.N = 96;
            c.K = 3 * nat::kNativeF16AccumBlockK + 7;
            c.threads = 4;
            c.bias_dt = kF16;
            const f16_operands_t o = operands_of(c, 3300);
            const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
            direct_t d = make_direct(c, 3301);
            const status_t st = nt::run_matmul(d.call);
            const std::string log = ::testing::internal::GetCapturedStdout();
            const std::string line_at = "[MATMUL.NATIVE_BRGEMM] M=9 N=96 ";
            const size_t at = log.find(line_at);
            const std::string line = at == std::string::npos
                    ? std::string()
                    : log.substr(at, log.find('\n', at) - at);
            const bool warned
                    = log.find(
                              "ZENDNNL_MATMUL_NATIVE_F16_ACCUM=bf16 is neither "
                              "f16 nor f32")
                    != std::string::npos;
            if (on_ph != s.ph || st != status_t::success
                    || line.find(s.ph ? " kernels=f16_ph " : " kernels=f16 ")
                            == std::string::npos
                    || (zendnnl::error_handling::apilog_warning_enabled()
                            && warned != s.warns)) {
                std::fprintf(stderr,
                        "%s: f16_ph %d (expected %d), status %d, warned %d, "
                        "line: %s\n",
                        what.c_str(), on_ph, s.ph, static_cast<int>(st), warned,
                        line.c_str());
                ++bad;
            }
            if (!same_bits(c, d.dst, model_of(d))) {
                std::fprintf(stderr, "%s: not the bits of the %s model\n",
                        what.c_str(), s.ph ? "blocked" : "K-order");
                ++bad;
            }
            if (!same_bits(c, run_native(c, o), model_output(c, o))) ++bad;
            return bad + (::testing::Test::HasFailure() ? 1 : 0);
        }, what);
    }
}

} // namespace
