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

// The FP32 GEMM kernels of the native GEMM (ZENDNNL_MATMUL_ALGO=10): the tail
// kernel over every row and column count it takes, the GEMM looper on a CPU
// without AVX-512, the scalar kernel's beta, and the calls whose beta / alpha
// the loopers cannot take.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/kernel/fp32/fp32_gemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/fp32_gemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"
#include "native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using nat::fused_postop_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

/// Rows and columns of B in the packed panels the tail kernel reads.
constexpr int kPanel = 64;

std::vector<float> random_floats(size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float &x : v)
        x = dist(gen);
    return v;
}

/// Fills the stack below the caller's frame with NaN, so that a kernel
/// called next reads NaN from any accumulator it did not write.
ZENDNNL_NOINLINE float poison_stack() {
    volatile float pad[8192];
    for (int i = 0; i < 8192; ++i)
        pad[i] = std::numeric_limits<float>::quiet_NaN();
    return pad[0];
}

/// C[m][n] = beta * C0[m][n] + sum_k A[m][k] B[k][n] (+ bias[n], then relu),
/// over rows of A K apart and rows of B kPanel apart.
void reference(const std::vector<float> &A, const std::vector<float> &B,
        const std::vector<float> &C0, const float *bias, int mr, int nr, int K,
        int ldc, float beta, bool relu, std::vector<double> &out,
        std::vector<double> &mag) {
    out.assign(static_cast<size_t>(mr) * ldc, 0.0);
    mag.assign(out.size(), 0.0);
    for (int m = 0; m < mr; ++m)
        for (int n = 0; n < nr; ++n) {
            double acc = 0.0, sum = 0.0;
            if (beta != 0.0f) {
                acc = static_cast<double>(beta) * C0[m * ldc + n];
                sum = std::fabs(acc);
            }
            for (int k = 0; k < K; ++k) {
                const double p
                        = static_cast<double>(A[m * K + k]) * B[k * kPanel + n];
                acc += p;
                sum += std::fabs(p);
            }
            if (bias != nullptr) {
                acc += bias[n];
                sum += std::fabs(bias[n]);
            }
            if (relu && acc < 0.0) acc = 0.0;
            out[m * ldc + n] = acc;
            mag[m * ldc + n] = sum;
        }
}

bool has_avx512() {
    return nat::detect_uarch().avx512f;
}

/// Every row count 1 to 12 and column counts around each vector boundary,
/// with beta 0, 1 and 0.5, with and without bias and relu: no output is
/// NaN (each accumulator is written before it is read) and each matches
/// the reference.
void check_tail_kernel() {
    const int K = 37;
    const std::vector<float> A = random_floats(12 * K, 11);
    const std::vector<float> B = random_floats(K * kPanel, 12);
    const std::vector<float> bias = random_floats(kPanel, 13);
    const int ldc = kPanel + 3;
    const int cols[] = {1, 5, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64};
    const float betas[] = {0.0f, 1.0f, 0.5f};
    std::vector<double> ref, mag;
    int cases = 0, failures = 0;
    for (int mr = 1; mr <= 12; ++mr)
        for (int nr : cols)
            for (float beta : betas)
                for (int variant = 0; variant < 2; ++variant) {
                    const bool with_bias = variant == 1;
                    const fused_postop_t op = with_bias ? fused_postop_t::relu
                                                        : fused_postop_t::none;
                    const std::vector<float> C0 = random_floats(
                            static_cast<size_t>(12) * ldc, 100 + mr * 64 + nr);
                    std::vector<float> C = C0;
                    const float *b = with_bias ? bias.data() : nullptr;
                    (void)poison_stack();
                    nat::avx512_tail_kernel(A.data(), K, B.data(), kPanel,
                            C.data(), ldc, K, mr, nr, beta, b, op);
                    reference(A, B, C0, b, mr, nr, K, ldc, beta, with_bias, ref,
                            mag);
                    ++cases;
                    bool ok = true;
                    for (int m = 0; m < 12 && ok; ++m)
                        for (int n = 0; n < ldc && ok; ++n) {
                            const size_t i = static_cast<size_t>(m) * ldc + n;
                            if (m >= mr || n >= nr) {
                                // Outside the tile: untouched.
                                ok = C[i] == C0[i];
                            } else {
                                ok = !std::isnan(C[i])
                                        && std::fabs(C[i] - ref[i])
                                                <= 1e-5 * mag[i] + 1e-6;
                            }
                        }
                    if (!ok) {
                        ++failures;
                        ADD_FAILURE()
                                << "gemm tail kernel: mr " << mr << " nr " << nr
                                << " beta " << beta << " bias " << with_bias;
                    }
                }
    EXPECT_EQ(failures, 0) << "of " << cases << " cases";
}

TEST(NativeFp32TailKernel, GemmTailWritesEveryAccumulatorBeforeReadingIt) {
    if (!has_avx512()) GTEST_SKIP() << "needs AVX-512";
    check_tail_kernel();
}

/// The CPU of a host without AVX-512: the detected parameters with every
/// AVX-512 feature cleared.
nat::UarchParams without_avx512() {
    nat::UarchParams u = nat::detect_uarch();
    u.avx512f = false;
    u.avx512bf16 = false;
    u.avx512vnni = false;
    return u;
}

/// An FP32 call of the native GEMM looper on uarch: dst = alpha * A * B +
/// beta * dst (+ bias), then ops; row-major, tight leading dimensions.
void run_fp32_gemm(const nat::UarchParams &uarch, int M, int N, int K,
        float alpha, float beta, const std::vector<float> &A,
        const std::vector<float> &B, const float *bias,
        const std::vector<mm::matmul_post_op> &ops, int threads,
        std::vector<float> &C) {
    nat::GemmDescriptor d;
    d.M = M;
    d.N = N;
    d.K = K;
    d.lda = K;
    d.ldb = N;
    d.ldc = N;
    d.alpha = alpha;
    d.beta = beta;
    d.bias_dt = bias != nullptr ? data_type_t::f32 : data_type_t::none;
    d.bias = bias;
    d.num_threads = threads;
    mm::matmul_params params;
    params.postop_ = ops;
    nat::gemm_execute(d, uarch, A.data(), B.data(), C.data(), bias, params);
}

mm::matmul_post_op eltwise(post_op_type_t type) {
    mm::matmul_post_op po;
    po.po_type = type;
    po.alpha = 1.0f;
    return po;
}

// Without AVX-512 the looper runs the scalar kernel and the portable
// epilogue: with alpha 1.5 (scaling, bias and post-ops after the kernel),
// each post-op is the portable formula of the output without it, bit for
// bit.  The AVX-512 epilogue computes these post-ops by approximations.
TEST(NativeFp32GemmIsa, WithoutAvx512TheEpilogueIsPortable) {
    const nat::UarchParams u = without_avx512();
    const int M = 13, N = 40, K = 24;
    const std::vector<float> A = random_floats(static_cast<size_t>(M) * K, 31);
    const std::vector<float> B = random_floats(static_cast<size_t>(K) * N, 32);
    const std::vector<float> bias = random_floats(N, 33);
    struct op_t {
        const char *name;
        post_op_type_t type;
        float (*f)(float);
    };
    const op_t ops[] = {
            {"gelu_erf", post_op_type_t::gelu_erf,
                    [](float v) {
        return 0.5f * v * (1.0f + std::erf(v * 0.7071067811865476f));
    }},
            {"gelu_tanh", post_op_type_t::gelu_tanh,
                    [](float v) {
        const float x3 = v * v * v;
        return 0.5f * v
                * (1.0f
                        + std::tanh(
                                0.7978845608028654f * (v + 0.044715f * x3)));
    }},
            {"sigmoid", post_op_type_t::sigmoid,
                    [](float v) { return 1.0f / (1.0f + std::exp(-v)); }},
            {"tanh", post_op_type_t::tanh,
                    [](float v) { return std::tanh(v); }},
            {"exp", post_op_type_t::exp, [](float v) { return std::exp(v); }},
            {"elu", post_op_type_t::elu,
                    [](float v) {
        return v < 0.0f ? 1.0f * (std::exp(v) - 1.0f) : v;
    }},
    };
    for (const int threads : {1, 4}) {
        std::vector<float> plain(static_cast<size_t>(M) * N, 0.0f);
        run_fp32_gemm(
                u, M, N, K, 1.5f, 0.0f, A, B, bias.data(), {}, threads, plain);
        for (const op_t &op : ops) {
            std::vector<float> got(plain.size(), 0.0f);
            run_fp32_gemm(u, M, N, K, 1.5f, 0.0f, A, B, bias.data(),
                    {eltwise(op.type)}, threads, got);
            int differ = 0;
            for (size_t i = 0; i < got.size(); ++i)
                differ += got[i] != op.f(plain[i]);
            EXPECT_EQ(differ, 0) << op.name << " threads " << threads << ": of "
                                 << got.size() << " elements";
        }
    }
}

// Without AVX-512 a binary post-op reads the operand at each element's row
// and column, before or after an activation.
TEST(NativeFp32GemmIsa, WithoutAvx512BinaryOperandsFollowTheTile) {
    const nat::UarchParams u = without_avx512();
    const int M = 13, N = 40, K = 24;
    const std::vector<float> A = random_floats(static_cast<size_t>(M) * K, 41);
    const std::vector<float> B = random_floats(static_cast<size_t>(K) * N, 42);
    std::vector<float> operand = random_floats(static_cast<size_t>(M) * N, 43);
    mm::matmul_post_op add;
    add.po_type = post_op_type_t::binary_add;
    add.buff = operand.data();
    add.dtype = data_type_t::f32;
    add.dims = {M, N};
    add.leading_dim = N;
    std::vector<float> plain(static_cast<size_t>(M) * N, 0.0f);
    run_fp32_gemm(u, M, N, K, 1.5f, 0.0f, A, B, nullptr, {}, 4, plain);
    std::vector<float> got(plain.size(), 0.0f);
    run_fp32_gemm(u, M, N, K, 1.5f, 0.0f, A, B, nullptr,
            {add, eltwise(post_op_type_t::tanh)}, 4, got);
    int differ = 0;
    for (size_t i = 0; i < got.size(); ++i)
        differ += got[i] != std::tanh(plain[i] + operand[i]);
    EXPECT_EQ(differ, 0) << "of " << got.size() << " elements";
}

const float kBetas[] = {0.0f, 1.0f, 0.5f, -1.25f, 2.0f};

/// |got - (alpha * A * B + beta * C0 + bias)| within the rounding of a
/// K-deep sum over every element of an M x N tile of row stride ldc, B of
/// row stride ldb.
::testing::AssertionResult matches(const std::vector<float> &got,
        const std::vector<float> &A, const std::vector<float> &B,
        const std::vector<float> &C0, const float *bias, int M, int N, int K,
        int ldb, int ldc, float alpha, float beta) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double ab = 0.0, mag = 0.0;
            for (int k = 0; k < K; ++k) {
                const double p
                        = static_cast<double>(A[m * K + k]) * B[k * ldb + n];
                ab += p;
                mag += std::fabs(p);
            }
            const size_t i = static_cast<size_t>(m) * ldc + n;
            double ref = alpha * ab;
            mag = std::fabs(alpha) * mag;
            if (beta != 0.0f) {
                ref += static_cast<double>(beta) * C0[i];
                mag += std::fabs(static_cast<double>(beta) * C0[i]);
            }
            if (bias != nullptr) {
                ref += bias[n];
                mag += std::fabs(bias[n]);
            }
            if (!(std::fabs(got[i] - ref) <= 1e-5 * mag + 1e-6))
                return ::testing::AssertionFailure()
                        << "row " << m << " col " << n << ": " << got[i]
                        << " for " << ref;
        }
    return ::testing::AssertionSuccess();
}

// The scalar kernel scales C by beta, as the AVX-512 kernels do: called
// with beta, and as the looper defers alpha, with beta / alpha before C is
// scaled by alpha.
TEST(NativeFp32ScalarKernel, ScalesCByBeta) {
    const int mr = 5, nr = 23, k = 31, ldc = 29;
    const std::vector<float> A = random_floats(static_cast<size_t>(mr) * k, 51);
    const std::vector<float> B = random_floats(static_cast<size_t>(k) * nr, 52);
    const std::vector<float> C0
            = random_floats(static_cast<size_t>(mr) * ldc, 53);
    for (const float beta : kBetas)
        for (const float alpha : {1.0f, 1.5f, -0.75f}) {
            const float kernel_beta
                    = alpha != 1.0f && beta != 0.0f ? beta / alpha : beta;
            std::vector<float> C = C0;
            nat::scalar_microkernel(A.data(), k, B.data(), nr, C.data(), ldc, k,
                    mr, nr, kernel_beta, nullptr, fused_postop_t::none);
            for (int m = 0; m < mr; ++m)
                for (int n = 0; n < nr; ++n)
                    C[m * ldc + n] *= alpha;
            EXPECT_TRUE(matches(
                    C, A, B, C0, nullptr, mr, nr, k, nr, ldc, alpha, beta))
                    << "beta " << beta << " alpha " << alpha;
            for (int m = 0; m < mr; ++m)
                for (int n = nr; n < ldc; ++n)
                    EXPECT_EQ(C[m * ldc + n], C0[m * ldc + n]);
        }
}

// Without AVX-512 the looper computes alpha * A * B + beta * C + bias for
// any alpha and beta of a finite beta / alpha.
TEST(NativeFp32ScalarKernel, LooperWithoutAvx512ScalesByAlphaAndBeta) {
    const nat::UarchParams u = without_avx512();
    const int M = 13, N = 40, K = 24;
    const std::vector<float> A = random_floats(static_cast<size_t>(M) * K, 61);
    const std::vector<float> B = random_floats(static_cast<size_t>(K) * N, 62);
    const std::vector<float> C0 = random_floats(static_cast<size_t>(M) * N, 63);
    const std::vector<float> bias = random_floats(N, 64);
    for (const int threads : {1, 4})
        for (const float beta : kBetas)
            for (const float alpha : {1.0f, 1.5f}) {
                std::vector<float> C = C0;
                run_fp32_gemm(u, M, N, K, alpha, beta, A, B, bias.data(), {},
                        threads, C);
                EXPECT_TRUE(matches(
                        C, A, B, C0, bias.data(), M, N, K, N, N, alpha, beta))
                        << "beta " << beta << " alpha " << alpha << " threads "
                        << threads;
            }
}

// The GEMM loopers pass beta / alpha to their kernels: the native GEMM does
// not take a call where it is not finite (alpha 0, or an alpha small enough
// for it to overflow) and writes nothing, and matmul_direct computes alpha *
// A * B + beta * C + bias for it on another route; the calls the native GEMM
// takes keep its results.
TEST(NativeGemmBetaOverAlpha, CallsWithoutAFiniteQuotientGoElsewhere) {
    const nat::UarchParams &u = nat::detect_uarch();
    const int M = 13, N = 64, K = 32;
    const std::vector<float> C0 = random_floats(static_cast<size_t>(M) * N, 71);
    const std::vector<float> bias = random_floats(N, 72);
    for (const data_type_t dt : {data_type_t::f32, data_type_t::bf16}) {
        if (dt == data_type_t::bf16 && !u.avx512bf16) continue;
        const nt::buffer_t a = nt::random_buffer(
                dt, static_cast<size_t>(M) * K, 73, -1.0f, 1.0f);
        const nt::buffer_t b = nt::random_buffer(
                dt, static_cast<size_t>(K) * N, 74, -1.0f, 1.0f);
        std::vector<float> A(static_cast<size_t>(M) * K);
        std::vector<float> B(static_cast<size_t>(K) * N);
        for (size_t i = 0; i < A.size(); ++i)
            A[i] = nt::load_f32(a.get(), dt, i);
        for (size_t i = 0; i < B.size(); ++i)
            B[i] = nt::load_f32(b.get(), dt, i);
        for (const float alpha :
                {0.0f, std::numeric_limits<float>::denorm_min(), 1.5f})
            for (const float beta : {0.0f, 0.5f, -1.25f}) {
                SCOPED_TRACE(
                        std::string(dt == data_type_t::f32 ? "f32" : "bf16")
                        + " alpha " + std::to_string(alpha) + " beta "
                        + std::to_string(beta));
                const bool taken = beta == 0.0f || std::isfinite(beta / alpha);
                mm::matmul_params p;
                p.dtypes.src = dt;
                p.dtypes.wei = dt;
                p.dtypes.dst = data_type_t::f32;
                p.dtypes.bias = data_type_t::f32;
                p.lowoha_algo = matmul_algo_t::native_gemm;
                p.num_threads = 1;
                std::vector<float> C = C0;
                mm::matmul_params q = p;
                EXPECT_EQ(nat::native_gemm_execute('r', false, false, M, N, K,
                                  alpha, a.get(), K, b.get(), N, bias.data(),
                                  beta, C.data(), N, false, 1, q),
                        taken);
                if (taken)
                    EXPECT_TRUE(matches(C, A, B, C0, bias.data(), M, N, K, N, N,
                            alpha, beta));
                else
                    EXPECT_EQ(C, C0) << "the native GEMM wrote dst";
                C = C0;
                mm::matmul_batch_params_t batch;
                ASSERT_EQ(mm::matmul_direct('r', false, false, M, N, K, alpha,
                                  a.get(), K, b.get(), N, bias.data(), beta,
                                  C.data(), N, false, batch, p),
                        status_t::success);
                EXPECT_TRUE(matches(
                        C, A, B, C0, bias.data(), M, N, K, N, N, alpha, beta));
            }
    }
}

} // namespace
