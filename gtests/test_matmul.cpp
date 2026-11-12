/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "common/platform_info.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/ggml_weight_unpack.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include <gtest/gtest.h>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

namespace {

/// The epsilon of the comparison of an f16 call of K steps on algo whose
/// products lie within max_ab of zero: epsilon, but on the native BRGEMM's
/// F16_PH kernels, which add the products of each block of
/// kNativeF16AccumBlockK steps in f16 (unit roundoff 2^-11).  There the
/// partial sums of a block of b steps of products of random sign stay near
/// sqrt(b) * max_ab, and their rounding errors, of random sign too, leave
/// an output within 4 * 2^-11 * sqrt(b * K) * max_ab of the exact sum (the
/// bound gamma(b, 2^-11) * sum |a b| holds for any signs); the comparison
/// takes K * epsilon for that term.
float f16_epsilon(matmul_algo_t algo, uint64_t K, float max_ab, float epsilon) {
    namespace nat = zendnnl::lowoha::matmul::native;
    if (algo != matmul_algo_t::native_brgemm
            || nat::native_kernel_registry().f16_kind()
                    != nat::native_kernel_kind_t::f16_ph)
        return epsilon;
    const double b = std::min<double>(
            static_cast<double>(K), nat::kNativeF16AccumBlockK);
    return epsilon
            + static_cast<float>(4.0 * 0x1p-11
                    * std::sqrt(b / static_cast<double>(K)) * max_ab);
}

/// The values of t, an f16 tensor, widened to f32 into like, an f32 tensor
/// of its size, order and strides; any other tensor as it is.
tensor_t widened_to_f32(tensor_t t, tensor_t like) {
    if (t.get_data_type() != data_type_t::f16) return t;
    const uint16_t *from
            = static_cast<const uint16_t *>(t.get_raw_handle_unsafe());
    float *to = static_cast<float *>(like.get_raw_handle_unsafe());
    const uint64_t n = t.get_buffer_sz_bytes() / sizeof(uint16_t);
    for (uint64_t i = 0; i < n; ++i)
        to[i] = zendnnl::common::float16_t::f16_to_f32_val(from[i]);
    return like;
}

/// The reference of an f16 call on a CPU without AVX512-FP16, where the
/// reference route takes no f16 operand: the reference over f32 copies of
/// the call's f16 operands (input32, weight32 and out32 f32 tensors of the
/// input's, the weight's and the output's geometry), its output rounded to
/// f16 into an f16 output.
status_t f16_reference_in_f32(tensor_factory_t &factory, tensor_t &input,
        tensor_t input32, tensor_t &weight, tensor_t weight32, tensor_t &bias,
        tensor_t &output, tensor_t out32,
        const std::vector<post_op_type_t> &po_types,
        const std::vector<tensor_t> &binary_tensors, bool use_LOWOHA,
        matmul_algo_t algo, float alpha, float beta) {
    tensor_t in = widened_to_f32(input, input32);
    tensor_t wei = widened_to_f32(weight, weight32);
    tensor_t b = widened_to_f32(bias,
            factory.uniform_dist_tensor(
                    bias.get_size(), data_type_t::f32, 1.0));
    std::vector<tensor_t> bin;
    for (const tensor_t &t : binary_tensors)
        bin.push_back(widened_to_f32(t,
                factory.uniform_dist_tensor(
                        t.get_size(), data_type_t::f32, 1.0)));
    tensor_t out = widened_to_f32(output, out32);
    const status_t st = matmul_kernel_test(in, wei, b, out, po_types, bin,
            use_LOWOHA, algo, alpha, beta, true);
    if (st == status_t::success && output.get_data_type() == data_type_t::f16) {
        const float *from
                = static_cast<const float *>(out.get_raw_handle_unsafe());
        uint16_t *to = static_cast<uint16_t *>(output.get_raw_handle_unsafe());
        const uint64_t n = output.get_buffer_sz_bytes() / sizeof(uint16_t);
        for (uint64_t i = 0; i < n; ++i)
            to[i] = zendnnl::common::float16_t::f32_to_f16_val(from[i]);
    }
    return st;
}

} // namespace

/** @brief TestMatmul is a test class to handle parameters */
class TestMatmul : public ::testing::TestWithParam<MatmulType> {
protected:
    /** @brief SetUp is to initialize test parameters
   *
   *  This method is a standard and is used in googletests to initialize parameters
   *  for each test and also acts as fixutres i.e. handling the common part of
   *  each test.
   *
   * */
    virtual void SetUp() {
        MatmulType params = GetParam();
        use_LOWOHA = params.use_LOWOHA;
        // LOWOHA-only mode: tests are masked when the user explicitly selects the
        // regular (non-LOWOHA) API. Skip with a message asking the user to use the
        // LOA (LOWOHA) API. Run this guard *before* any global side effects
        // (e.g. srand(), omp_set_num_threads) so skipped tests don't mutate
        // process state (RNG, OMP thread count) that subsequent test suites
        // rely on.
        if (!use_LOWOHA) {
            GTEST_SKIP() << "Skipping: please use LOA (LOWOHA) API. "
                         << "Omit --lowoha or pass --lowoha true to run these "
                            "tests.";
        }
        srand(static_cast<unsigned int>(seed));
        m = params.matmul_m;
        k = params.matmul_k;
        n = params.matmul_n;
        transA = params.transA;
        transB = params.transB;
        alpha = params.alpha;
        beta = params.beta;
        source_dtype = params.source_dtype;
        output_dtype = params.output_dtype;
        weight_granularity = params.weight_granularity;
        po_types = params.po_types;
        algo = params.algo;
        num_threads = params.num_threads;
        omp_set_num_threads(num_threads);
        log_info("m: ", m, " k: ", k, " n: ", n, " TransA: ", transA,
                " TransB: ", transB, " alpha: ", alpha, " beta: ", beta,
                " postops: ", postOpTypesToStr(po_types),
                " algo: ", static_cast<int>(algo),
                " num_threads: ", num_threads);
    }

    /** @brief TearDown is used to free resource used in test */
    virtual void TearDown() { clear_matmul_test_caches(); }
    uint64_t m, k, n;
    std::vector<post_op_type_t> po_types;
    bool transA, transB;
    tensor_factory_t tensor_factory {};
    float alpha, beta;
    bool use_LOWOHA;
    data_type_t source_dtype, output_dtype;
    quant_granularity_t weight_granularity;
    matmul_algo_t algo;
    int32_t num_threads;
};

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize parameters
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate matmul F32 aocl kernel support wrt Reference kernel
 */
TEST_P(TestMatmul, F32_F32) {
    auto weight_tensor = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::f32, 2.0, transB);
    auto input_tensor = tensor_factory.uniform_dist_tensor(
            {m, k}, data_type_t::f32, 2.0, transA);
    auto bias_tensor
            = tensor_factory.uniform_dist_tensor({1, n}, data_type_t::f32, 2.0);

    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);

    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, alpha, beta, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, alpha, beta);
    if (native_refusal_verified()) return;
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);
    bool enable_f32_relaxation = (algo == matmul_algo_t::libxsmm
            || algo == matmul_algo_t::libxsmm_blocked
            || algo == matmul_algo_t::native_brgemm);
    if (is_test_successful) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol_f32, epsilon_f32, is_test_successful,
                enable_f32_relaxation, alpha, false, beta);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param WOQ_BF16_S4 user-defined name of test according to test
 *  @brief Test to validate matmul WOQ (Weight-Only Quantization) with BF16 input
 *         and S4 weights wrt Reference kernel
 */
TEST_P(TestMatmul, WOQ_BF16_S4) {
    if (algo == matmul_algo_t::onednn
            || algo == matmul_algo_t::onednn_blocked) {
        GTEST_SKIP()
                << "WOQ (bf16xs4) is unsupported on the oneDNN matmul backend; "
                   "only AOCL-DLP implements the WOQ path.";
    }
    // Test WOQ with different scale/zp granularity combinations:
    // Combination 0: scale=per-tensor - {1,1}
    // Combination 1: scale=per-channel - {1,n}
    // Combination 2: scale=per-group - {G,n}

    int quant_combo = rand() % 3;

    std::vector<uint64_t> scale_size;
    uint64_t group_size = 0;
    uint64_t num_groups = 1;
    std::string scale_granularity;

    // Calculate group size for per-group cases
    // DLP only supports even group sizes
    // Find all even divisors of K as valid group sizes
    std::vector<uint64_t> valid_group_sizes;
    for (uint64_t gs = 2; gs <= k; gs += 2) { // Only even numbers
        if (k % gs == 0) { valid_group_sizes.push_back(gs); }
    }
    // If no even divisors found, fall back to K only if K is even
    // Otherwise, fall back to per-channel (combo 1) for per-group cases
    bool has_valid_group_size = !valid_group_sizes.empty() || (k % 2 == 0);
    if (valid_group_sizes.empty() && k % 2 == 0) {
        valid_group_sizes.push_back(k);
    }
    // If K is odd and no even divisors, downgrade per-group to per-channel
    if (!has_valid_group_size && quant_combo == 2) {
        quant_combo = 1; // Fall back to per-channel
    }

    switch (quant_combo) {
        case 0:
            // scale=per-tensor
            scale_size = {1, 1};
            scale_granularity = "per-tensor";
            break;

        case 1:
            // scale=per-channel
            scale_size = {1, n};
            scale_granularity = "per-channel";
            break;

        case 2: {
            // scale=per-group
            // Randomly select one of the valid group sizes (safe here since we checked has_valid_group_size)
            uint64_t valid_group_size
                    = valid_group_sizes[rand() % valid_group_sizes.size()];
            group_size = valid_group_size;
            num_groups = k / group_size;
            scale_size = {num_groups, n};
            scale_granularity = "per-group";
            break;
        }
    }

    auto scale_dtype = (rand() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;
    auto wei_scale
            = tensor_factory.uniform_dist_tensor(scale_size, scale_dtype, 2.0);

    // Log test configuration
    std::string group_info = quant_combo == 2
            ? " group_size=" + std::to_string(group_size)
                    + " num_groups=" + std::to_string(num_groups)
            : "";
    log_info("WOQ test: scale=", scale_granularity, "[", scale_size[0], ",",
            scale_size[1], "]",
            " scale_dtype=", (scale_dtype == data_type_t::f32 ? "f32" : "bf16"),
            group_info);

    // Create S4 quantized weight tensor with scale
    auto weight_tensor = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::s4, 7.0, transB, wei_scale);

    // BF16 input tensor
    auto input_tensor = tensor_factory.uniform_dist_tensor(
            {m, k}, data_type_t::bf16, 2.0, transA);

    // Bias tensor (optional, can be BF16 or F32)
    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);

    // Binary tensor for post-ops if needed
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);

    auto output_dtype = rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32;
    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, output_dtype, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, output_dtype, 2.0);

    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, alpha, beta, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, alpha, beta);
    if (native_refusal_verified()) return;

    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                output_dtype == data_type_t::bf16 ? rtol_bf16 : rtol_woq,
                output_dtype == data_type_t::bf16 ? epsilon_bf16 : epsilon_woq,
                is_test_successful, false, alpha, true, beta);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param W4A8 user-defined name of test according to test
 *  @brief W4A8: dynamic bf16 src to s8, s4 wei, vs reference kernel
 */
TEST_P(TestMatmul, W4A8_BF16) {
    if (!use_LOWOHA) {
        GTEST_SKIP() << "W4A8 lives in run_dlp(), reachable only via "
                        "the LOWOHA path.";
    }
    if (algo == matmul_algo_t::onednn
            || algo == matmul_algo_t::onednn_blocked) {
        GTEST_SKIP();
    }

    // AOCL s8s8_sym_quant requires K-group size % 4 == 0.
    uint64_t sym_k = (k / 4) * 4;
    if (sym_k == 0) { sym_k = 4; }

    std::mt19937 local_rng(m ^ k ^ n ^ 0xDA8C);
    std::vector<uint64_t> valid_gs;
    for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
        if (sym_k % gs == 0) { valid_gs.push_back(gs); }
    }
    if (valid_gs.empty()) {
        GTEST_SKIP() << "No valid W4A8 group_size for K=" << sym_k;
    }
    uint64_t group_size = valid_gs[local_rng() % valid_gs.size()];
    uint64_t num_groups = sym_k / group_size;

    std::vector<uint64_t> src_scale_size;
    std::vector<uint64_t> wei_scale_size = {num_groups, n};
    std::string src_granularity_label;
    if (local_rng() % 2 == 0) {
        src_scale_size = {m, 1};
        src_granularity_label = "per-token";
    } else {
        src_scale_size = {m, num_groups};
        src_granularity_label = "per-group";
    }
    std::string granularity_label
            = "A=" + src_granularity_label + " + W=per-group";

    neutralize_mish_quant_int8(po_types);

    data_type_t ref_dt = data_type_t::bf16;
    data_type_t scale_dt
            = (local_rng() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;

    auto wei_scale
            = tensor_factory.uniform_dist_tensor(wei_scale_size, scale_dt, 2.0);
    auto weight_tensor = tensor_factory.uniform_dist_tensor(
            {sym_k, n}, data_type_t::s4, 7.0, transB, wei_scale);

    auto src_scale = tensor_factory.zero_tensor(src_scale_size, scale_dt);
    auto input_tensor = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, ref_dt, 2.0, transA, src_scale, tensor_t());
    auto src_ref = input_tensor;

    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);

    data_type_t out_dt = data_type_t::bf16;
    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);

    log_info("W4A8: m=", m, " sym_k=", sym_k, " n=", n,
            " granularity=", granularity_label, " group_size=", group_size,
            " num_groups=", num_groups, " out_dt=bf16",
            " scale_dt=", scale_dt == data_type_t::f32 ? "f32" : "bf16");

    status_t ref_status = matmul_kernel_test(src_ref, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0f, 0.0f, true);

    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0f, 0.0f);
    if (native_refusal_verified()) return;

    EXPECT_EQ(status, status_t::success)
            << "W4A8: kernel path returned failure "
               "(expected success on dynamic bf16 src + s4 wei via aocl_dlp / "
               "aocl_dlp_blocked)";
    EXPECT_EQ(ref_status, status_t::success)
            << "W4A8: WOQ reference path (bf16 src + s4 wei) returned failure";

    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                rtol_bf16, 128 * epsilon_bf16, ok,
                /*enable_f32_relaxation=*/false, /*alpha=*/1.0f,
                /*is_quant=*/true);
    }
    EXPECT_TRUE(ok);
}

TEST_P(TestMatmul, W4A8_F32) {
    if (!use_LOWOHA) {
        GTEST_SKIP() << "W4A8 lives in run_dlp(), reachable only via "
                        "the LOWOHA path.";
    }
    if (algo == matmul_algo_t::onednn
            || algo == matmul_algo_t::onednn_blocked) {
        GTEST_SKIP();
    }

    uint64_t sym_k = (k / 4) * 4;
    if (sym_k == 0) { sym_k = 4; }

    std::mt19937 local_rng(m ^ k ^ n ^ 0xDA8C);
    std::vector<uint64_t> valid_gs;
    for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
        if (sym_k % gs == 0) { valid_gs.push_back(gs); }
    }
    if (valid_gs.empty()) {
        GTEST_SKIP() << "No valid W4A8 group_size for K=" << sym_k;
    }
    uint64_t group_size = valid_gs[local_rng() % valid_gs.size()];
    uint64_t num_groups = sym_k / group_size;

    std::vector<uint64_t> src_scale_size;
    std::vector<uint64_t> wei_scale_size = {num_groups, n};
    std::string src_granularity_label;
    if (local_rng() % 2 == 0) {
        src_scale_size = {m, 1};
        src_granularity_label = "per-token";
    } else {
        src_scale_size = {m, num_groups};
        src_granularity_label = "per-group";
    }
    std::string granularity_label
            = "A=" + src_granularity_label + " + W=per-group";

    neutralize_mish_quant_int8(po_types);

    data_type_t ref_dt = data_type_t::bf16;
    data_type_t scale_dt
            = (local_rng() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;

    auto wei_scale
            = tensor_factory.uniform_dist_tensor(wei_scale_size, scale_dt, 2.0);
    auto weight_tensor = tensor_factory.uniform_dist_tensor(
            {sym_k, n}, data_type_t::s4, 7.0, transB, wei_scale);

    auto src_scale = tensor_factory.zero_tensor(src_scale_size, scale_dt);
    auto input_tensor = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, ref_dt, 2.0, transA, src_scale, tensor_t());
    auto src_ref = input_tensor;

    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);

    constexpr data_type_t out_dt = data_type_t::f32;
    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);

    log_info("W4A8: m=", m, " sym_k=", sym_k, " n=", n,
            " granularity=", granularity_label, " group_size=", group_size,
            " num_groups=", num_groups, " out_dt=f32",
            " scale_dt=", scale_dt == data_type_t::f32 ? "f32" : "bf16");

    status_t ref_status = matmul_kernel_test(src_ref, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0f, 0.0f, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0f, 0.0f);
    if (native_refusal_verified()) return;

    EXPECT_EQ(status, status_t::success)
            << "W4A8 F32: kernel path returned failure";
    EXPECT_EQ(ref_status, status_t::success)
            << "W4A8 F32: reference path returned failure";

    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                rtol_woq, 128 * epsilon_woq, ok,
                /*enable_f32_relaxation=*/false, /*alpha=*/1.0f,
                /*is_quant=*/true);
    }
    EXPECT_TRUE(ok);
}

TEST(W4A8Direct, DynamicF32SourcePassesValidationAndExecutes) {
    constexpr int M = 2;
    constexpr int N = 32;
    constexpr int K = 32;

    std::vector<float> src(static_cast<size_t>(M) * K, 1.0f);
    // transB=true: packed signed-S4 [N,K], two +1 nibbles per byte.
    std::vector<int8_t> weight(
            static_cast<size_t>(N) * K / 2, static_cast<int8_t>(0x11));
    std::vector<float> wei_scale(N, 1.0f);
    std::vector<float> dst(static_cast<size_t>(M) * N, -1.0f);

    matmul_params params;
    params.dtypes.src = data_type_t::f32;
    params.dtypes.wei = data_type_t::s4;
    params.dtypes.dst = data_type_t::f32;
    params.dtypes.bias = data_type_t::none;
    params.dtypes.compute = data_type_t::s8;
    params.dynamic_quant = true;
    params.weight_cache_type = 1;
    params.num_threads = 4;
    params.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;
    params.quant_params.src_scale.dt = data_type_t::f32;
    params.quant_params.src_scale.dims = {M, 1};
    params.quant_params.wei_scale.buff = wei_scale.data();
    params.quant_params.wei_scale.dt = data_type_t::f32;
    params.quant_params.wei_scale.dims = {1, N};

    ASSERT_EQ(validate_matmul_direct_inputs(src.data(), weight.data(),
                      dst.data(), M, N, K, /*Batch_A=*/1, /*Batch_B=*/1, params,
                      /*is_weights_const=*/true),
            status_t::success);

    matmul_batch_params_t batch;
    const status_t status = run_matmul_direct('r', /*transA=*/false,
            /*transB=*/true, M, N, K, 1.0f, src.data(), K, weight.data(), K,
            nullptr, 0.0f, dst.data(), N, true, batch, params);
    if (native_refusal_verified()) return;
    ASSERT_EQ(status, status_t::success);
    for (float value : dst) {
        EXPECT_NEAR(value, static_cast<float>(K), 0.1f);
    }
}

/** @brief W4A8 static: pre-quantized s8 src + s4 wei vs WOQ bf16 reference.
 *
 *  quant_params_compute writes per-group src scales in physical {G,M} order
 *  when the bf16 source is transposed. matmul_direct consumes logical {M,G},
 *  matching the dynamic W4A8 untranspose in reorder_quantization_wrapper.
 */
static void untranspose_logical_mg_src_scale(
        tensor_t &scale, int64_t M_dim, int64_t G_dim) {
    if (M_dim <= 1 || G_dim <= 1) { return; }
    const size_t nelems
            = static_cast<size_t>(M_dim) * static_cast<size_t>(G_dim);
    const size_t elem_size = zendnnl::common::size_of(scale.get_data_type());
    uint8_t *buf = static_cast<uint8_t *>(scale.get_raw_handle_unsafe());
    if (!buf) { return; }
    std::vector<uint8_t> tmp(nelems * elem_size);
    std::memcpy(tmp.data(), buf, nelems * elem_size);
    for (int64_t row = 0; row < M_dim; ++row) {
        for (int64_t col = 0; col < G_dim; ++col) {
            std::memcpy(
                    buf + static_cast<size_t>(row * G_dim + col) * elem_size,
                    tmp.data()
                            + static_cast<size_t>(col * M_dim + row)
                                    * elem_size,
                    elem_size);
        }
    }
}

TEST_P(TestMatmul, W4A8_STATIC_S8) {
    if (!use_LOWOHA) {
        GTEST_SKIP() << "W4A8 lives in run_dlp(), reachable only via "
                        "the LOWOHA path.";
    }
    if (algo == matmul_algo_t::onednn
            || algo == matmul_algo_t::onednn_blocked) {
        GTEST_SKIP();
    }

    uint64_t sym_k = (k / 4) * 4;
    if (sym_k == 0) { sym_k = 4; }

    std::mt19937 local_rng(m ^ k ^ n ^ 0x5A71);
    std::vector<uint64_t> valid_gs;
    for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
        if (sym_k % gs == 0) { valid_gs.push_back(gs); }
    }
    if (valid_gs.empty()) {
        GTEST_SKIP() << "No valid W4A8 group_size for K=" << sym_k;
    }
    uint64_t group_size = valid_gs[local_rng() % valid_gs.size()];
    uint64_t num_groups = sym_k / group_size;

    std::vector<uint64_t> src_scale_size;
    std::vector<uint64_t> wei_scale_size = {num_groups, n};
    // transA + per-group {M,G} is the layout that must stay logical for GEMM.
    if (transA || local_rng() % 2 != 0) {
        src_scale_size = {m, num_groups};
    } else {
        src_scale_size = {m, 1};
    }

    neutralize_mish_quant_int8(po_types);

    data_type_t ref_dt = data_type_t::bf16;
    data_type_t scale_dt
            = (local_rng() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;

    auto wei_scale
            = tensor_factory.uniform_dist_tensor(wei_scale_size, scale_dt, 2.0);
    auto weight_tensor = tensor_factory.uniform_dist_tensor(
            {sym_k, n}, data_type_t::s4, 7.0, transB, wei_scale);

    auto src_scale = tensor_factory.zero_tensor(src_scale_size, scale_dt);
    auto src_ref = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, ref_dt, 2.0, transA, src_scale, tensor_t());

    tensor_t input_tensor;
    std::vector<int64_t> src_sd(src_scale_size.begin(), src_scale_size.end());
    tensor_t src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                src_sd, scale_dt, src_scale, src_zp, &input_tensor)
            != status_t::success) {
        FAIL() << "static W4A8 source quantization failed";
    }
    if (transA && src_scale_size.size() == 2 && src_scale_size[1] > 1) {
        untranspose_logical_mg_src_scale(src_scale,
                static_cast<int64_t>(src_scale_size[0]),
                static_cast<int64_t>(src_scale_size[1]));
    }

    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);

    auto output_tensor = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);

    status_t ref_status = matmul_kernel_test(src_ref, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0f, 0.0f, true);

    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0f, 0.0f);
    if (native_refusal_verified()) return;

    EXPECT_EQ(status, status_t::success)
            << "W4A8 static: kernel path (s8 src + s4 wei) failed";
    EXPECT_EQ(ref_status, status_t::success)
            << "W4A8 static: WOQ bf16 reference path failed";

    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                rtol_bf16, 128 * epsilon_bf16, ok,
                /*enable_f32_relaxation=*/false, /*alpha=*/1.0f,
                /*is_quant=*/true);
    }
    EXPECT_TRUE(ok);

    matmul_params dq_reject;
    dq_reject.dtypes.src = data_type_t::s8;
    dq_reject.dtypes.wei = data_type_t::s4;
    dq_reject.dtypes.dst = data_type_t::bf16;
    dq_reject.dtypes.compute = data_type_t::s8;
    dq_reject.dynamic_quant = true;
    dq_reject.quant_params.src_scale.buff = src_scale.get_raw_handle_unsafe();
    dq_reject.quant_params.src_scale.dt = scale_dt;
    dq_reject.quant_params.src_scale.dims = src_sd;
    dq_reject.quant_params.wei_scale.buff = wei_scale.get_raw_handle_unsafe();
    dq_reject.quant_params.wei_scale.dt = scale_dt;
    dq_reject.quant_params.wei_scale.dims.assign(
            wei_scale_size.begin(), wei_scale_size.end());
    EXPECT_EQ(
            validate_matmul_direct_inputs(input_tensor.get_raw_handle_unsafe(),
                    weight_tensor.get_raw_handle_unsafe(),
                    output_tensor.get_raw_handle_unsafe(), static_cast<int>(m),
                    static_cast<int>(n), static_cast<int>(sym_k), 1, 1,
                    dq_reject,
                    /*is_weights_const=*/true),
            status_t::failure);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param WOQ_BF16_U4 user-defined name of test according to test
 *  @brief Test to validate matmul WOQ (Weight-Only Quantization) with BF16 input
 *         and U4 weights wrt Reference kernel
 */
TEST_P(TestMatmul, WOQ_BF16_U4) {
    // Test WOQ with different scale/zp granularity combinations:
    // Combination 0: scale=per-tensor,  zp=per-tensor  -> {1,1}, {1,1}
    // Combination 1: scale=per-channel, zp=per-tensor  -> {1,n}, {1,1}
    // Combination 2: scale=per-channel,   zp=per-channel  -> {1,n}, {1,n}
    // Combination 3: scale=per-group,   zp=per-group   -> {G,n}, {G,n}

    int quant_combo = (rand() + k) % 4;
    bool random_zp_domain = (rand() + k) % 2 == 0;

    std::vector<uint64_t> scale_size;
    std::vector<uint64_t> zp_size;
    uint64_t group_size = 0;
    uint64_t num_groups = 1;
    std::string scale_granularity, zp_granularity;

    // Calculate group size for per-group cases
    // DLP only supports even group sizes
    // Find all even divisors of K as valid group sizes
    std::vector<uint64_t> valid_group_sizes;
    for (uint64_t gs = 2; gs <= k; gs += 2) { // Only even numbers
        if (k % gs == 0) { valid_group_sizes.push_back(gs); }
    }
    // If no even divisors found, fall back to K only if K is even
    // Otherwise, fall back to per-channel (combo 1) for per-group cases
    bool has_valid_group_size = !valid_group_sizes.empty() || (k % 2 == 0);
    if (valid_group_sizes.empty() && k % 2 == 0) {
        valid_group_sizes.push_back(k);
    }
    // If K is odd and no even divisors, downgrade per-group to per-channel
    if (!has_valid_group_size && (quant_combo == 2 || quant_combo == 3)) {
        quant_combo = 1; // Fall back to per-channel
    }

    switch (quant_combo) {
        case 0:
            // scale=per-tensor, zp=per-tensor
            scale_size = {1, 1};
            zp_size = {1, 1};
            scale_granularity = "per-tensor";
            zp_granularity = "per-tensor";
            break;

        case 1:
            // scale=per-channel, zp=per-tensor
            scale_size = {1, n};
            zp_size = {1, 1};
            scale_granularity = "per-channel";
            zp_granularity = "per-tensor";
            break;

        case 2:
            // scale=per-channel, zp=per-channel
            scale_size = {1, n};
            zp_size = {1, n};
            scale_granularity = "per-channel";
            zp_granularity = "per-channel";
            break;

        case 3: {
            // scale=per-group, zp=per-group
            // Randomly select one of the valid group sizes (safe here since we checked has_valid_group_size)
            uint64_t valid_group_size
                    = valid_group_sizes[rand() % valid_group_sizes.size()];
            group_size = valid_group_size;
            num_groups = k / group_size;
            scale_size = {num_groups, n};
            zp_size = {num_groups, n};
            scale_granularity = "per-group";
            zp_granularity = "per-group";
            break;
        }
    }

    auto scale_dtype = (rand() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;
    auto wei_scale
            = tensor_factory.uniform_dist_tensor(scale_size, scale_dtype, 2.0);
    auto wei_zp = tensor_factory.uniform_dist_tensor(zp_size,
            random_zp_domain ? data_type_t::bf16 : data_type_t::s8,
            random_zp_domain ? 2.0 : 25.0);

    // Log test configuration
    std::string group_info = quant_combo > 2
            ? " group_size=" + std::to_string(group_size)
                    + " num_groups=" + std::to_string(num_groups)
            : "";
    log_info("WOQ test: scale=", scale_granularity, "[", scale_size[0], ",",
            scale_size[1], "]", " zp=", zp_granularity, "[", zp_size[0], ",",
            zp_size[1], "]",
            " scale_dtype=", (scale_dtype == data_type_t::f32 ? "f32" : "bf16"),
            group_info);

    // Create U4 quantized weight tensor with scale and zero point
    auto weight_tensor = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::u4, 15.0, transB, wei_scale, wei_zp);

    // BF16 input tensor
    auto input_tensor = tensor_factory.uniform_dist_tensor(
            {m, k}, data_type_t::bf16, 2.0, transA);

    // Bias tensor (optional, can be BF16 or F32)
    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);

    // Binary tensor for post-ops if needed
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);

    auto output_dtype = rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32;
    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, output_dtype, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, output_dtype, 2.0);

    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, alpha, beta, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, alpha, beta);
    if (native_refusal_verified()) return;

    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                output_dtype == data_type_t::bf16 ? rtol_bf16 : rtol_woq,
                output_dtype == data_type_t::bf16 ? epsilon_bf16 : epsilon_woq,
                is_test_successful, false, alpha, true, beta);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param BF16_F32 user-defined name of test according to test
 *  @brief Test to validate matmul BF16outputF32 aocl kernel support wrt Reference kernel
 */
TEST_P(TestMatmul, BF16_F32) {
    auto weight_tensor = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::bf16, 2.0, transB);
    auto input_tensor = tensor_factory.uniform_dist_tensor(
            {m, k}, data_type_t::bf16, 2.0, transA);
    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
    // Run the reference kernel first. The optimized path may reorder weights
    // in the caller's buffer; the golden kernel must see the original layout.
    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, alpha, beta, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, alpha, beta);
    if (native_refusal_verified()) return;
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);
    bool enable_f32_relaxation = (algo == matmul_algo_t::libxsmm
            || algo == matmul_algo_t::libxsmm_blocked
            || algo == matmul_algo_t::native_brgemm);
    if (is_test_successful) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol_f32, epsilon_f32, is_test_successful,
                enable_f32_relaxation, alpha, false, beta);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param BF16_BF16 user-defined name of test according to test
 *  @brief Test to validate matmul BF16outputBF16 aocl kernel support wrt Reference kernel
 */
TEST_P(TestMatmul, BF16_BF16) {
    // TODO: Extend support for test cases with a wider range of values.
    auto weight_tensor = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::bf16, 2.0, transB);
    auto input_tensor = tensor_factory.uniform_dist_tensor(
            {m, k}, data_type_t::bf16, 2.0, transA);
    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);
    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, alpha, beta, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, alpha, beta);
    if (native_refusal_verified()) return;

    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol_bf16, epsilon_bf16, is_test_successful, false, alpha,
                false, beta);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param F16_F16 user-defined name of test according to test
 *  @brief Test to validate matmul F16 input/weight/output
 */
TEST_P(TestMatmul, F16_F16) {
    auto bias_dtype
            = (rand() + k) % 2 == 0 ? data_type_t::f16 : data_type_t::f32;
    auto binary_dtype
            = (rand() + k) % 2 == 0 ? data_type_t::f16 : data_type_t::f32;
    auto weight_tensor = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::f16, 2.0, transB);
    auto input_tensor = tensor_factory.uniform_dist_tensor(
            {m, k}, data_type_t::f16, 2.0, transA);
    auto bias_tensor
            = tensor_factory.uniform_dist_tensor({1, n}, bias_dtype, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape, binary_dtype);
    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f16, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f16, 2.0);
    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, alpha, beta, true);
    if (ref_status == status_t::isa_unsupported
            && !zendnnl_platform_info().get_avx512_f16_status())
        ref_status = f16_reference_in_f32(tensor_factory, input_tensor,
                tensor_factory.uniform_dist_tensor(
                        {m, k}, data_type_t::f32, 1.0, transA),
                weight_tensor,
                tensor_factory.uniform_dist_tensor(
                        {k, n}, data_type_t::f32, 1.0, transB),
                bias_tensor, output_tensor_ref,
                tensor_factory.uniform_dist_tensor(
                        {m, n}, data_type_t::f32, 1.0),
                po_types, binary_tensors, use_LOWOHA, algo, alpha, beta);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, alpha, beta);
    if (native_refusal_verified()) return;
    if (status == status_t::isa_unsupported) {
        GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        // Products of values within 2 of zero.
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol_bf16, f16_epsilon(algo, k, 4.0f, epsilon_bf16),
                is_test_successful, false, alpha, false, beta);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param F16_F32 user-defined name of test according to test
 *  @brief Test to validate matmul F16 input/weight with F32 output
 */
TEST_P(TestMatmul, F16_F32) {
    auto bias_dtype
            = (rand() + k) % 2 == 0 ? data_type_t::f16 : data_type_t::f32;
    auto binary_dtype
            = (rand() + k) % 2 == 0 ? data_type_t::f16 : data_type_t::f32;
    auto weight_tensor = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::f16, 2.0, transB);
    auto input_tensor = tensor_factory.uniform_dist_tensor(
            {m, k}, data_type_t::f16, 2.0, transA);
    auto bias_tensor
            = tensor_factory.uniform_dist_tensor({1, n}, bias_dtype, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape, binary_dtype);
    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, alpha, beta, true);
    if (ref_status == status_t::isa_unsupported
            && !zendnnl_platform_info().get_avx512_f16_status())
        ref_status = f16_reference_in_f32(tensor_factory, input_tensor,
                tensor_factory.uniform_dist_tensor(
                        {m, k}, data_type_t::f32, 1.0, transA),
                weight_tensor,
                tensor_factory.uniform_dist_tensor(
                        {k, n}, data_type_t::f32, 1.0, transB),
                bias_tensor, output_tensor_ref, output_tensor_ref, po_types,
                binary_tensors, use_LOWOHA, algo, alpha, beta);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, alpha, beta);
    if (native_refusal_verified()) return;
    if (status == status_t::isa_unsupported) {
        GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        const float rtol = (algo == matmul_algo_t::onednn
                                   || algo == matmul_algo_t::onednn_blocked)
                ? rtol_f32
                : rtol_bf16;
        const float epsilon = (algo == matmul_algo_t::onednn
                                      || algo == matmul_algo_t::onednn_blocked)
                ? epsilon_f32
                : epsilon_bf16;
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol, epsilon, is_test_successful, false, alpha, false, beta);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmulStride parameterized test class to initialize parameters
 *  @param F32_F32 user-defined name of test
 *  @brief Test to validate matmul F32 aocl kernel support wrt Reference kernel
 *  with strided tensors.
 *
 */
TEST_P(TestMatmul, F32_F32_Stride) {
    size_t stride_in_inc = rand() % 50;
    size_t stride_wt_inc = rand() % 50;
    size_t stride_dst_inc = rand() % 50;
    std::vector<size_t> stride_in = {m, k};
    std::vector<size_t> stride_wt = {k, n};
    std::vector<size_t> stride_dst = {m, n + stride_dst_inc};
    if (transB) {
        stride_wt[0] += stride_wt_inc;
    } else {
        stride_wt[1] += stride_wt_inc;
    }
    if (transA) {
        stride_in[0] += stride_in_inc;
    } else {
        stride_in[1] += stride_in_inc;
    }
    auto weight_tensor = tensor_factory.uniform_dist_strided_tensor(
            {k, n}, stride_wt, data_type_t::f32, 2.0, transB);
    auto input_tensor = tensor_factory.uniform_dist_strided_tensor(
            {m, k}, stride_in, data_type_t::f32, 2.0, transA);
    auto bias_tensor
            = tensor_factory.uniform_dist_tensor({1, n}, data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor = tensor_factory.uniform_dist_strided_tensor(
            {m, n}, stride_dst, data_type_t::f32, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_strided_tensor(
            {m, n}, stride_dst, data_type_t::f32, 2.0);

    log_info("transA:", transA, " transB:", transB, " strided_inp:{",
            stride_in[0], ",", stride_in[1], "} strided_wt:{", stride_wt[0],
            ",", stride_wt[1], "}");

    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, alpha, beta, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, alpha, beta);
    if (native_refusal_verified()) return;
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);
    bool enable_f32_relaxation = (algo == matmul_algo_t::libxsmm
            || algo == matmul_algo_t::libxsmm_blocked
            || algo == matmul_algo_t::native_brgemm);
    if (is_test_successful) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol_f32, epsilon_f32, is_test_successful,
                enable_f32_relaxation, alpha, false, beta);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmulStride parameterized test class to initialize parameters
 *  @param BF16_F32 user-defined name of test
 *  @brief Test to validate matmul BF16 input, F32 output aocl kernel support
 *  wrt Reference kernel with strided tensors.
 *
 */
TEST_P(TestMatmul, BF16_F32_Stride) {
    size_t stride_in_inc = rand() % 50;
    size_t stride_wt_inc = rand() % 50;
    size_t stride_dst_inc = rand() % 50;
    std::vector<size_t> stride_in = {m, k};
    std::vector<size_t> stride_wt = {k, n};
    std::vector<size_t> stride_dst = {m, n + stride_dst_inc};
    if (transB) {
        stride_wt[0] += stride_wt_inc;
    } else {
        stride_wt[1] += stride_wt_inc;
    }
    if (transA) {
        stride_in[0] += stride_in_inc;
    } else {
        stride_in[1] += stride_in_inc;
    }
    auto weight_tensor = tensor_factory.uniform_dist_strided_tensor(
            {k, n}, stride_wt, data_type_t::bf16, 2.0, transB);
    auto input_tensor = tensor_factory.uniform_dist_strided_tensor(
            {m, k}, stride_in, data_type_t::bf16, 2.0, transA);
    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor = tensor_factory.uniform_dist_strided_tensor(
            {m, n}, stride_dst, data_type_t::f32, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_strided_tensor(
            {m, n}, stride_dst, data_type_t::f32, 2.0);

    log_info("transA:", transA, " transB:", transB, " strided_inp:{",
            stride_in[0], ",", stride_in[1], "} strided_wt:{", stride_wt[0],
            ",", stride_wt[1], "}");

    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, alpha, beta, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, alpha, beta);
    if (native_refusal_verified()) return;
    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    bool enable_f32_relaxation = (algo == matmul_algo_t::libxsmm
            || algo == matmul_algo_t::libxsmm_blocked
            || algo == matmul_algo_t::native_brgemm);
    if (is_test_successful) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol_f32, epsilon_f32, is_test_successful,
                enable_f32_relaxation, alpha, false, beta);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmulStride parameterized test class to initialize parameters
 *  @param BF16_BF16 user-defined name of test
 *  @brief Test to validate matmul BF16 input, BF16 output aocl kernel support
 *  wrt Reference kernel with strided tensors.
 *
 */
TEST_P(TestMatmul, BF16_BF16_Stride) {
    size_t stride_in_inc = rand() % 50;
    size_t stride_wt_inc = rand() % 50;
    size_t stride_dst_inc = rand() % 50;
    std::vector<size_t> stride_in = {m, k};
    std::vector<size_t> stride_wt = {k, n};
    std::vector<size_t> stride_dst = {m, n + stride_dst_inc};
    if (transB) {
        stride_wt[0] += stride_wt_inc;
    } else {
        stride_wt[1] += stride_wt_inc;
    }
    if (transA) {
        stride_in[0] += stride_in_inc;
    } else {
        stride_in[1] += stride_in_inc;
    }
    auto weight_tensor = tensor_factory.uniform_dist_strided_tensor(
            {k, n}, stride_wt, data_type_t::bf16, 2.0, transB);
    auto input_tensor = tensor_factory.uniform_dist_strided_tensor(
            {m, k}, stride_in, data_type_t::bf16, 2.0, transA);
    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor = tensor_factory.uniform_dist_strided_tensor(
            {m, n}, stride_dst, data_type_t::bf16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_strided_tensor(
            {m, n}, stride_dst, data_type_t::bf16, 2.0);

    log_info("transA:", transA, " transB:", transB, " strided_inp:{",
            stride_in[0], ",", stride_in[1], "} strided_wt:{", stride_wt[0],
            ",", stride_wt[1], "}");

    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, alpha, beta, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, alpha, beta);
    if (native_refusal_verified()) return;

    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol_bf16, epsilon_bf16, is_test_successful, false, alpha,
                false, beta);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmulStride parameterized test class to initialize parameters
 *  @param F16_F16_Stride user-defined name of test
 *  @brief Test to validate matmul F16 input, F16 output kernel support
 *  wrt Reference kernel with strided tensors.
 *
 */
TEST_P(TestMatmul, F16_F16_Stride) {
    size_t stride_in_inc = rand() % 50;
    size_t stride_wt_inc = rand() % 50;
    size_t stride_dst_inc = rand() % 50;
    std::vector<size_t> stride_in = {m, k};
    std::vector<size_t> stride_wt = {k, n};
    std::vector<size_t> stride_dst = {m, n + stride_dst_inc};
    if (transB) {
        stride_wt[0] += stride_wt_inc;
    } else {
        stride_wt[1] += stride_wt_inc;
    }
    if (transA) {
        stride_in[0] += stride_in_inc;
    } else {
        stride_in[1] += stride_in_inc;
    }
    auto bias_dtype
            = (rand() + k) % 2 == 0 ? data_type_t::f16 : data_type_t::f32;
    auto binary_dtype
            = (rand() + k) % 2 == 0 ? data_type_t::f16 : data_type_t::f32;
    auto weight_tensor = tensor_factory.uniform_dist_strided_tensor(
            {k, n}, stride_wt, data_type_t::f16, 2.0, transB);
    auto input_tensor = tensor_factory.uniform_dist_strided_tensor(
            {m, k}, stride_in, data_type_t::f16, 2.0, transA);
    auto bias_tensor
            = tensor_factory.uniform_dist_tensor({1, n}, bias_dtype, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape, binary_dtype);
    auto output_tensor = tensor_factory.uniform_dist_strided_tensor(
            {m, n}, stride_dst, data_type_t::f16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_strided_tensor(
            {m, n}, stride_dst, data_type_t::f16, 2.0);

    log_info("transA:", transA, " transB:", transB, " strided_inp:{",
            stride_in[0], ",", stride_in[1], "} strided_wt:{", stride_wt[0],
            ",", stride_wt[1], "}");

    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, alpha, beta, true);
    if (ref_status == status_t::isa_unsupported
            && !zendnnl_platform_info().get_avx512_f16_status())
        ref_status = f16_reference_in_f32(tensor_factory, input_tensor,
                tensor_factory.uniform_dist_strided_tensor(
                        {m, k}, stride_in, data_type_t::f32, 1.0, transA),
                weight_tensor,
                tensor_factory.uniform_dist_strided_tensor(
                        {k, n}, stride_wt, data_type_t::f32, 1.0, transB),
                bias_tensor, output_tensor_ref,
                tensor_factory.uniform_dist_strided_tensor(
                        {m, n}, stride_dst, data_type_t::f32, 1.0),
                po_types, binary_tensors, use_LOWOHA, algo, alpha, beta);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, alpha, beta);
    if (native_refusal_verified()) return;
    if (status == status_t::isa_unsupported) {
        GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }

    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        // Products of values within 2 of zero.
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol_bf16, f16_epsilon(algo, k, 4.0f, epsilon_bf16),
                is_test_successful, false, alpha, false, beta);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param INT8 user-defined name of test according to test
 *  @brief Test to validate matmul INT8 aocl kernel support wrt Reference kernel
 */
TEST_P(TestMatmul, INT8) {
    // TODO: Extend support for test cases with a wider range of values.
    std::vector<uint64_t> wei_scale_size
            = (weight_granularity == quant_granularity_t::tensor)
            ? std::vector<uint64_t> {1, 1}
            : std::vector<uint64_t> {1, n};

    data_type_t ref_dt = (output_dtype == data_type_t::f32) ? data_type_t::f32
                                                            : data_type_t::bf16;

    auto wei_ref
            = tensor_factory.uniform_dist_tensor({k, n}, ref_dt, 25.0, transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                {static_cast<int64_t>(wei_scale_size[0]),
                        static_cast<int64_t>(wei_scale_size[1])},
                data_type_t::f32, wei_scale, wei_zp, &weight_tensor)
            != status_t::success) {
        FAIL() << "weight dynamic quantization failed";
    }

    auto src_ref
            = tensor_factory.uniform_dist_tensor({m, k}, ref_dt, 25.0, transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, source_dtype,
                {1, 1}, data_type_t::f32, src_scale, src_zp, &input_tensor)
            != status_t::success) {
        FAIL() << "source dynamic quantization failed";
    }
    tensor_t dst_scale, dst_zp;
    // Destination scale/zp are only meaningful when the output is quantized
    // to an integer type (s8/u8). For floating-point outputs (f32/bf16/f16)
    // the matmul writes raw values and no dst-quant params are needed.
    if (output_dtype == data_type_t::s8 || output_dtype == data_type_t::u8) {
        auto dst_ref = tensor_factory.uniform_dist_tensor({m, n}, ref_dt, 2.0);
        if (quant_params_compute(tensor_factory, dst_ref, ref_dt, output_dtype,
                    {1, 1}, data_type_t::f32, dst_scale, dst_zp)
                != status_t::success) {
            FAIL() << "destination scale/zp computation failed";
        }
    }
    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor = tensor_factory.uniform_dist_tensor(
            {m, n}, output_dtype, 2.0, false, dst_scale, dst_zp);
    auto output_tensor_ref = tensor_factory.uniform_dist_tensor(
            {m, n}, output_dtype, 2.0, false, dst_scale, dst_zp);
    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0, 0.0, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0, 0.0);
    if (native_refusal_verified()) return;
    if (status == status_t::isa_unsupported) {
        GTEST_SKIP() << "F16 not supported: requires F16-capable ISA";
    }

    bool is_test_successful
            = (status == status_t::success && ref_status == status_t::success);

    if (is_test_successful) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol_bf16, epsilon_bf16, is_test_successful, false, 1.0f, true);
    }

    EXPECT_TRUE(is_test_successful);
}

/** @brief Test INT8 sym_quant: per-group source scale, bf16 output */
TEST_P(TestMatmul, INT8_SYM_QUANT_PER_GROUP_BF16) {
    uint64_t sym_k = (k / 4) * 4;
    if (sym_k == 0) { sym_k = 4; }
    std::vector<uint64_t> valid_gs;
    for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
        if (sym_k % gs == 0) { valid_gs.push_back(gs); }
    }
    if (valid_gs.empty()) {
        GTEST_SKIP() << "No valid group_size for K=" << sym_k;
    }
    std::mt19937 local_rng(m ^ k ^ n ^ 0xBF16);
    uint64_t group_size = valid_gs[local_rng() % valid_gs.size()];

    source_dtype = data_type_t::s8;
    use_LOWOHA = true;
    neutralize_mish_quant_int8(po_types);

    data_type_t ref_dt = data_type_t::bf16;
    data_type_t scale_dt
            = (local_rng() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;
    uint64_t num_groups = sym_k / group_size;
    std::vector<int64_t> wei_sd
            = {static_cast<int64_t>(num_groups), static_cast<int64_t>(n)};
    std::vector<int64_t> src_sd
            = {static_cast<int64_t>(m), static_cast<int64_t>(num_groups)};

    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {sym_k, n}, ref_dt, 25.0, transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor)
            != status_t::success) {
        FAIL() << "weight dynamic quantization failed";
    }

    auto src_ref = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, ref_dt, 25.0, transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                src_sd, scale_dt, src_scale, src_zp, &input_tensor)
            != status_t::success) {
        FAIL() << "source dynamic quantization failed";
    }

    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);

    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0, 0.0, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0, 0.0);
    if (native_refusal_verified()) return;
    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                rtol_bf16, epsilon_bf16, ok, false, 1.0f, true);
    }
    EXPECT_TRUE(ok);
}

/** @brief Test INT8 sym_quant: per-group source scale, f32 output */
TEST_P(TestMatmul, INT8_SYM_QUANT_PER_GROUP_F32) {
    uint64_t sym_k = (k / 4) * 4;
    if (sym_k == 0) { sym_k = 4; }
    std::vector<uint64_t> valid_gs;
    for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
        if (sym_k % gs == 0) { valid_gs.push_back(gs); }
    }
    if (valid_gs.empty()) {
        GTEST_SKIP() << "No valid group_size for K=" << sym_k;
    }
    std::mt19937 local_rng(m ^ k ^ n ^ 0xF320);
    uint64_t group_size = valid_gs[local_rng() % valid_gs.size()];

    source_dtype = data_type_t::s8;
    use_LOWOHA = true;
    neutralize_mish_quant_int8(po_types);

    data_type_t ref_dt = data_type_t::f32;
    data_type_t scale_dt
            = (local_rng() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;
    uint64_t num_groups = sym_k / group_size;
    std::vector<int64_t> wei_sd
            = {static_cast<int64_t>(num_groups), static_cast<int64_t>(n)};
    std::vector<int64_t> src_sd
            = {static_cast<int64_t>(m), static_cast<int64_t>(num_groups)};
    // Keeping the range to 2.0 to avoid accuracy drops in INT8 matmul
    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {sym_k, n}, ref_dt, 2.0, transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor)
            != status_t::success) {
        FAIL() << "weight dynamic quantization failed";
    }

    auto src_ref = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, ref_dt, 25.0, transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                src_sd, scale_dt, src_scale, src_zp, &input_tensor)
            != status_t::success) {
        FAIL() << "source dynamic quantization failed";
    }

    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);

    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);

    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0, 0.0, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0, 0.0);
    if (native_refusal_verified()) return;
    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                rtol_f32, epsilon_f32, ok, false, 1.0f, true);
    }
    EXPECT_TRUE(ok);
}

/** @brief Test INT8 sym_quant with GGML Q8_0 packed weights.
 *  Combines INT8_SYM_QUANT_PER_GROUP_BF16 and F32 variants:
 *    - Randomly picks bf16 or f32 for both input/output
 *    - K is always a multiple of 32 (GGML Q8_0 group size)
 *    - Weights are repacked into GGML Q8_0 blocked format and passed
 *      with pack_format_b = 1 so the API unpacks them internally
 */
TEST_P(TestMatmul, INT8_PER_GROUP_GGML_PACKED) {
    uint64_t sym_k = (k / 32) * 32;
    if (sym_k < 64) { sym_k = 64; }

    std::mt19937 local_rng(m ^ k ^ n ^ 0xBB01);
    bool use_bf16 = (local_rng() % 2 == 0);
    data_type_t ref_dt = use_bf16 ? data_type_t::bf16 : data_type_t::f32;
    data_type_t out_dt = ref_dt;

    source_dtype = data_type_t::s8;
    use_LOWOHA = true;
    neutralize_mish_quant_int8(po_types);

    uint64_t num_groups = sym_k / 32;
    data_type_t scale_dt = data_type_t::bf16;

    std::vector<int64_t> wei_sd
            = {static_cast<int64_t>(num_groups), static_cast<int64_t>(n)};
    std::vector<int64_t> src_sd
            = {static_cast<int64_t>(m), static_cast<int64_t>(num_groups)};

    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {sym_k, n}, ref_dt, 2.0, false);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor)
            != status_t::success) {
        FAIL() << "weight dynamic quantization failed";
    }

    auto src_ref = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, ref_dt, 25.0, transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                src_sd, scale_dt, src_scale, src_zp, &input_tensor)
            != status_t::success) {
        FAIL() << "source dynamic quantization failed";
    }

    const int8_t *raw_wt = static_cast<const int8_t *>(
            weight_tensor.get_raw_handle_unsafe());

    int64_t M_pack = static_cast<int64_t>(n);
    int64_t K_pack = static_cast<int64_t>(sym_k);
    int64_t ng = K_pack / 32;
    size_t num_scales = static_cast<size_t>(num_groups * n);

    const auto *raw_scl_bf16
            = static_cast<const uint16_t *>(wei_scale.get_raw_handle_unsafe());
    std::vector<float> scl_f32(num_scales);
    for (size_t i = 0; i < num_scales; i++) {
        uint32_t bits = static_cast<uint32_t>(raw_scl_bf16[i]) << 16;
        std::memcpy(&scl_f32[i], &bits, sizeof(float));
    }

    std::vector<int8_t> wt_nk(sym_k * n);
    for (uint64_t ki = 0; ki < sym_k; ki++) {
        for (uint64_t ni = 0; ni < n; ni++) {
            wt_nk[ni * sym_k + ki] = raw_wt[ki * n + ni];
        }
    }

    size_t packed_size = static_cast<size_t>(M_pack * ng * 34);
    std::vector<uint8_t> packed_buf(packed_size);
    repack_weights_q8_0(
            wt_nk.data(), scl_f32.data(), M_pack, K_pack, packed_buf.data());

    auto packed_weight_tensor = tensor_factory.copy_tensor({sym_k, n},
            data_type_t::s8,
            std::make_pair(packed_size, static_cast<void *>(packed_buf.data())),
            true, false);
    std::vector<uint8_t> packed_before(packed_size);
    std::memcpy(packed_before.data(),
            packed_weight_tensor.get_raw_handle_unsafe(), packed_size);

    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);

    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);

    log_info("INT8_SYM_QUANT_GGML_PACKED: dtype=", use_bf16 ? "bf16" : "f32",
            " K=", sym_k, " groups=", num_groups);

    status_t status = matmul_kernel_test(input_tensor, packed_weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0, 0.0, false, 1);
    if (native_refusal_verified()) return;
    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0, 0.0, true);

    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                rtol_bf16, epsilon_bf16, ok, false, 1.0f, true);
    }
    EXPECT_EQ(
            std::memcmp(packed_before.data(),
                    packed_weight_tensor.get_raw_handle_unsafe(), packed_size),
            0)
            << "GGML packed weights should not be modified by matmul_direct";
    EXPECT_TRUE(ok);
}

/** @brief Test INT8 sym_quant with GGML Q4_0 (4-bit) packed weights.
 *  Mirrors INT8_PER_GROUP_GGML_PACKED but packs weights into GGML Q4_0
 *  (block_q4_0) format and passes them as s4 with pack_format_b = 1, so the API
 *  unpacks the 4-bit weights to packed signed nibbles and runs the s8×s4
 *  per-group sym-quant GEMM.  The reference consumes the same codes as s8.
 *  Weights are clamped to the 4-bit range [-8, 7] so the packed source and the
 *  unpacked result are bit-identical to the reference.
 */
TEST_P(TestMatmul, INT8_PER_GROUP_GGML_PACKED_Q4_0) {
    uint64_t sym_k = (k / 32) * 32;
    if (sym_k < 64) { sym_k = 64; }

    std::mt19937 local_rng(m ^ k ^ n ^ 0xC401);
    bool use_bf16 = (local_rng() % 2 == 0);
    data_type_t ref_dt = use_bf16 ? data_type_t::bf16 : data_type_t::f32;
    data_type_t out_dt = ref_dt;

    source_dtype = data_type_t::s8;
    use_LOWOHA = true;
    neutralize_mish_quant_int8(po_types);

    uint64_t num_groups = sym_k / 32;
    data_type_t scale_dt = data_type_t::bf16;

    std::vector<int64_t> wei_sd
            = {static_cast<int64_t>(num_groups), static_cast<int64_t>(n)};
    std::vector<int64_t> src_sd
            = {static_cast<int64_t>(m), static_cast<int64_t>(num_groups)};

    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {sym_k, n}, ref_dt, 2.0, false);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor)
            != status_t::success) {
        FAIL() << "weight dynamic quantization failed";
    }

    // Overwrite the quantized weight with values spanning the FULL signed 4-bit
    // range [-8, 7] so every nibble code — including the sign-bit boundary
    // (-8 = 0b1000) and the max (+7) — is exercised on every shape/dtype combo.
    // The reference consumes these same s8 values, so the Q4_0 pack/unpack is a
    // lossless round-trip and any nibble-decode or sign-extension error surfaces
    // as an output mismatch.  The stride 7 is coprime with 16, so the pattern
    // cycles through all 16 codes and de-correlates from the 32-wide group edge.
    auto *wt_mut = static_cast<int8_t *>(weight_tensor.get_raw_handle_unsafe());
    for (uint64_t i = 0; i < sym_k * n; i++) {
        wt_mut[i] = static_cast<int8_t>(static_cast<int>((i * 7 + 3) % 16) - 8);
    }

    auto src_ref = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, ref_dt, 25.0, transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                src_sd, scale_dt, src_scale, src_zp, &input_tensor)
            != status_t::success) {
        FAIL() << "source dynamic quantization failed";
    }

    const int8_t *raw_wt = static_cast<const int8_t *>(
            weight_tensor.get_raw_handle_unsafe());

    int64_t M_pack = static_cast<int64_t>(n);
    int64_t K_pack = static_cast<int64_t>(sym_k);
    int64_t ng = K_pack / 32;
    size_t num_scales = static_cast<size_t>(num_groups * n);

    const auto *raw_scl_bf16
            = static_cast<const uint16_t *>(wei_scale.get_raw_handle_unsafe());
    std::vector<float> scl_f32(num_scales);
    for (size_t i = 0; i < num_scales; i++) {
        uint32_t bits = static_cast<uint32_t>(raw_scl_bf16[i]) << 16;
        std::memcpy(&scl_f32[i], &bits, sizeof(float));
    }

    std::vector<int8_t> wt_nk(sym_k * n);
    for (uint64_t ki = 0; ki < sym_k; ki++) {
        for (uint64_t ni = 0; ni < n; ni++) {
            wt_nk[ni * sym_k + ki] = raw_wt[ki * n + ni];
        }
    }

    // block_q4_0 = 2-byte fp16 scale + 16 packed-nibble bytes = 18 bytes.
    size_t packed_size = static_cast<size_t>(M_pack * ng * 18);
    std::vector<uint8_t> packed_buf(packed_size);
    repack_weights_q4_0(
            wt_nk.data(), scl_f32.data(), M_pack, K_pack, packed_buf.data());

    auto packed_weight_tensor = tensor_factory.copy_tensor({sym_k, n},
            data_type_t::s4,
            std::make_pair(packed_size, static_cast<void *>(packed_buf.data())),
            true, false);
    std::vector<uint8_t> packed_before(packed_size);
    std::memcpy(packed_before.data(),
            packed_weight_tensor.get_raw_handle_unsafe(), packed_size);

    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);

    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);

    log_info("INT8_SYM_QUANT_GGML_PACKED_Q4_0: dtype=",
            use_bf16 ? "bf16" : "f32", " K=", sym_k, " groups=", num_groups);

    status_t status = matmul_kernel_test(input_tensor, packed_weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0, 0.0, false, 1);
    if (native_refusal_verified()) return;
    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0, 0.0, true);

    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                rtol_bf16, epsilon_bf16, ok, false, 1.0f, true);
    }
    EXPECT_EQ(
            std::memcmp(packed_before.data(),
                    packed_weight_tensor.get_raw_handle_unsafe(), packed_size),
            0)
            << "GGML Q4_0 packed weights should not be modified by "
               "matmul_direct";
    EXPECT_TRUE(ok);
}

/** @brief Same as INT8_PER_GROUP_GGML_PACKED_Q4_0 with block_q4_0x8 weights.
 *  params.packing.is_superblock selects the 8-row interleaved layout.  Rows
 *  that are not a multiple of 8 cannot be a superblock, so those shapes skip.
 */
TEST_P(TestMatmul, INT8_PER_GROUP_GGML_PACKED_Q4_0_SUPERBLOCK) {
    if (n % 8 != 0) {
        GTEST_SKIP() << "Q4_0 superblock requires N divisible by 8";
    }
    uint64_t sym_k = (k / 32) * 32;
    if (sym_k < 64) { sym_k = 64; }

    std::mt19937 local_rng(m ^ k ^ n ^ 0xC408);
    bool use_bf16 = (local_rng() % 2 == 0);
    data_type_t ref_dt = use_bf16 ? data_type_t::bf16 : data_type_t::f32;
    data_type_t out_dt = ref_dt;

    source_dtype = data_type_t::s8;
    use_LOWOHA = true;
    neutralize_mish_quant_int8(po_types);

    uint64_t num_groups = sym_k / 32;
    data_type_t scale_dt = data_type_t::bf16;

    std::vector<int64_t> wei_sd
            = {static_cast<int64_t>(num_groups), static_cast<int64_t>(n)};
    std::vector<int64_t> src_sd
            = {static_cast<int64_t>(m), static_cast<int64_t>(num_groups)};

    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {sym_k, n}, ref_dt, 2.0, false);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor)
            != status_t::success) {
        FAIL() << "weight dynamic quantization failed";
    }

    auto *wt_mut = static_cast<int8_t *>(weight_tensor.get_raw_handle_unsafe());
    for (uint64_t i = 0; i < sym_k * n; i++) {
        wt_mut[i] = static_cast<int8_t>(static_cast<int>((i * 7 + 3) % 16) - 8);
    }

    auto src_ref = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, ref_dt, 25.0, transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                src_sd, scale_dt, src_scale, src_zp, &input_tensor)
            != status_t::success) {
        FAIL() << "source dynamic quantization failed";
    }

    const int8_t *raw_wt = static_cast<const int8_t *>(
            weight_tensor.get_raw_handle_unsafe());

    int64_t M_pack = static_cast<int64_t>(n);
    int64_t K_pack = static_cast<int64_t>(sym_k);
    int64_t ng = K_pack / 32;
    size_t num_scales = static_cast<size_t>(num_groups * n);

    const auto *raw_scl_bf16
            = static_cast<const uint16_t *>(wei_scale.get_raw_handle_unsafe());
    std::vector<float> scl_f32(num_scales);
    for (size_t i = 0; i < num_scales; i++) {
        uint32_t bits = static_cast<uint32_t>(raw_scl_bf16[i]) << 16;
        std::memcpy(&scl_f32[i], &bits, sizeof(float));
    }

    std::vector<int8_t> wt_nk(sym_k * n);
    for (uint64_t ki = 0; ki < sym_k; ki++) {
        for (uint64_t ni = 0; ni < n; ni++) {
            wt_nk[ni * sym_k + ki] = raw_wt[ki * n + ni];
        }
    }

    // block_q4_0x8 = 16-byte fp16 scales + 128 nibble bytes = 144 bytes,
    // (N/8) blocks per group.  That is the same byte count as plain Q4_0.
    size_t packed_size = static_cast<size_t>((M_pack / 8) * ng * 144);
    std::vector<uint8_t> packed_buf(packed_size);
    repack_weights_q4_0_superblock(
            wt_nk.data(), scl_f32.data(), M_pack, K_pack, packed_buf.data());

    auto packed_weight_tensor = tensor_factory.copy_tensor({sym_k, n},
            data_type_t::s4,
            std::make_pair(packed_size, static_cast<void *>(packed_buf.data())),
            true, false);
    std::vector<uint8_t> packed_before(packed_size);
    std::memcpy(packed_before.data(),
            packed_weight_tensor.get_raw_handle_unsafe(), packed_size);

    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);

    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, out_dt, 2.0);

    log_info("INT8_SYM_QUANT_GGML_PACKED_Q4_0_SUPERBLOCK: dtype=",
            use_bf16 ? "bf16" : "f32", " K=", sym_k, " groups=", num_groups);

    status_t status = matmul_kernel_test(input_tensor, packed_weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0, 0.0, false, 1, /*is_superblock=*/true);
    if (native_refusal_verified()) return;
    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0, 0.0, true);

    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                rtol_bf16, epsilon_bf16, ok, false, 1.0f, true);
    }
    EXPECT_EQ(
            std::memcmp(packed_before.data(),
                    packed_weight_tensor.get_raw_handle_unsafe(), packed_size),
            0)
            << "GGML Q4_0 superblock weights should not be modified by "
               "matmul_direct";
    EXPECT_TRUE(ok);
}

// Sign-extend one packed nibble.  Index is the [N, K] element; byte index/2
// holds columns {2j, 2j+1} in the low and high nibbles.
static int packed_s4_code(const uint8_t *bytes, int index) {
    const uint8_t byte = bytes[static_cast<size_t>(index) >> 1];
    const int nib = (index & 1) ? static_cast<int>(byte >> 4)
                                : static_cast<int>(byte & 0x0F);
    return (nib & 8) ? (nib - 16) : nib;
}

/** @brief Focused, deterministic bit-exactness check of the GGML Q4_0 unpack.
 *
 *  Independent of the GEMM: it packs a known [N, K] weight covering the full
 *  signed 4-bit range into GGML Q4_0, unpacks it to packed signed nibbles,
 *  and asserts every code round-trips in the correct [N, K] position.  This
 *  localises any nibble-order or sign-extension error to the unpack layer
 *  (vs the broader INT8_PER_GROUP_GGML_PACKED_Q4_0 end-to-end comparison).
 */
TEST(GgmlWeightUnpackQ4_0, PackedS4BitExactAllNibbleCodes) {
    using zendnnl::lowoha::matmul::ggml_unpack_weight_buffer;
    using zendnnl::lowoha::matmul::ggml_unpack_weight_buffer_size;

    for (int K : {32, 64, 128}) { // one, two, and four GGML groups
        const int N = 6; // output channels
        const int ng = K / 32;

        // Reference [N, K] s8 weights spanning every signed 4-bit code [-8, 7];
        // stride 7 (coprime with 16) hits all codes incl. the -8 sign boundary.
        std::vector<int8_t> wnk(static_cast<size_t>(N) * K);
        for (int i = 0; i < N * K; ++i) {
            wnk[i] = static_cast<int8_t>(
                    static_cast<int>((i * 7 + 3) % 16) - 8);
        }
        // Per-group scales laid out column-major [g*N + row] (repack contract).
        std::vector<float> scl(static_cast<size_t>(ng) * N);
        for (int i = 0; i < ng * N; ++i)
            scl[i] = 0.125f * static_cast<float>(i + 1);

        // Pack -> GGML Q4_0 (block_q4_0 = 18 bytes).
        std::vector<uint8_t> packed(static_cast<size_t>(N) * ng * 18);
        repack_weights_q4_0(wnk.data(), scl.data(), N, K, packed.data());

        // Unpack Q4_0 -> packed signed s4 [N, K/2].
        const int64_t bufsz = ggml_unpack_weight_buffer_size(
                2, /*use_bf16_scales=*/true, N, K);
        ASSERT_GT(bufsz, 0) << "K=" << K;
        std::vector<uint8_t> ubuf(static_cast<size_t>(bufsz));
        int8_t *packed_s4 = nullptr;
        void *scales_out = nullptr;
        ASSERT_EQ(ggml_unpack_weight_buffer(packed.data(), /*ggml_type=*/2,
                          /*use_bf16_scales=*/true, N, K, &packed_s4,
                          &scales_out, ubuf.data()),
                0)
                << "K=" << K;

        const auto *nibbles = reinterpret_cast<const uint8_t *>(packed_s4);
        for (int i = 0; i < N * K; ++i) {
            ASSERT_EQ(packed_s4_code(nibbles, i), static_cast<int>(wnk[i]))
                    << "Q4_0 nibble round-trip mismatch at index " << i
                    << " (K=" << K << ")";
        }
    }
}

/** @brief End-to-end GGML Q4_0 unpack: packed blob stays packed signed s4.
 *
 *  Covers the whole `unpack_ggml_weights_and_cache` path rather than the
 *  decode alone, so the +8 bias removal is checked — get it wrong and every
 *  code lands 8 away from its true value.  The unpack hands back packed
 *  nibbles + {K/32, N} bf16 scales (no s8 widen, no AOCL blocking).
 */
TEST(GgmlWeightUnpackQ4_0, CachedUnpackKeepsPackedS4) {
    const int N = 6;
    const int K = 64;
    const int ng = K / 32;

    std::vector<int8_t> wnk(static_cast<size_t>(N) * K);
    for (int i = 0; i < N * K; ++i) {
        wnk[i] = static_cast<int8_t>(static_cast<int>((i * 7 + 3) % 16) - 8);
    }
    std::vector<float> scl(static_cast<size_t>(ng) * N);
    for (int i = 0; i < ng * N; ++i)
        scl[i] = 0.125f * static_cast<float>(i + 1);

    std::vector<uint8_t> packed(static_cast<size_t>(N) * ng * 18);
    repack_weights_q4_0(wnk.data(), scl.data(), N, K, packed.data());

    clear_ggml_weight_unpack_cache();
    matmul_params params;
    params.dtypes.wei = data_type_t::s4;
    params.packing.pack_format_b = 1;

    const void *weight = static_cast<const void *>(packed.data());
    ASSERT_EQ(
            unpack_ggml_weights_and_cache(weight, N, K, /*ldb=*/K, 't', params),
            status_t::success);

    EXPECT_EQ(params.weight_cache_type, 2);
    EXPECT_EQ(params.packing.pack_format_b, 0);
    EXPECT_EQ(params.dtypes.wei, data_type_t::s4);
    EXPECT_GE(params.wei_buffer_capacity_bytes,
            static_cast<size_t>(N) * static_cast<size_t>(K / 2));
    ASSERT_NE(params.quant_params.wei_scale.buff, nullptr);
    EXPECT_EQ(params.quant_params.wei_scale.buff,
            static_cast<const void *>(static_cast<const uint8_t *>(weight)
                    + params.wei_buffer_capacity_bytes));

    const auto *nibbles = static_cast<const uint8_t *>(weight);
    for (int i = 0; i < N * K; ++i) {
        ASSERT_EQ(packed_s4_code(nibbles, i), static_cast<int>(wnk[i]))
                << "Q4_0 packed-s4 mismatch at index " << i;
    }
    clear_ggml_weight_unpack_cache();
}

/** @brief Q4_0 capacity covers the packed nibbles.  The s8 custom-kernel
 *  pack is not reserved: Q4_0 stays s4 and reorders through s8×s4.
 */
TEST(GgmlWeightUnpackQ4_0, UnpackCapacityCoversPackedS4) {
    const int N = 64; // divisible by kNRMin=32
    const int K = 256;
    const int ng = K / 32;

    std::vector<int8_t> wnk(static_cast<size_t>(N) * K, 1);
    std::vector<float> scl(static_cast<size_t>(ng) * N, 0.125f);
    std::vector<uint8_t> packed(static_cast<size_t>(N) * ng * 18);
    repack_weights_q4_0(wnk.data(), scl.data(), N, K, packed.data());

    clear_ggml_weight_unpack_cache();
    matmul_params params;
    params.dtypes.wei = data_type_t::s4;
    params.packing.pack_format_b = 1;
    const void *weight = static_cast<const void *>(packed.data());
    ASSERT_EQ(
            unpack_ggml_weights_and_cache(weight, N, K, /*ldb=*/K, 't', params),
            status_t::success);

    const size_t packed_bytes
            = static_cast<size_t>(N) * static_cast<size_t>(K / 2);
    EXPECT_EQ(params.packing.pack_format_b, 0);
    EXPECT_EQ(params.weight_cache_type, 2);
    EXPECT_EQ(params.dtypes.wei, data_type_t::s4);
    EXPECT_GE(params.wei_buffer_capacity_bytes, packed_bytes);
    clear_ggml_weight_unpack_cache();
}

/** @brief Report Llama-2-7B FFN unpack capacity + RSS for Q4_0 and Q8_0.
 *
 *  Q8_0 capacity covers the s8 image and the CK INT8 pack.  Q4_0 stays
 *  packed s4, so its capacity covers the nibble image (and the s8×s4
 *  blocked reorder), not an N*K s8 widen.  Scales sit past that capacity.
 */
TEST(GgmlWeightUnpackMem, Llama7bFfnUnpackCapacityAndRss) {
    using zendnnl::lowoha::matmul::clear_ggml_weight_unpack_cache;
    using zendnnl::lowoha::matmul::unpack_ggml_weights_and_cache;

    const int N = 11008;
    const int K = 4096;
    const int ng = K / 32;
    const size_t plain = static_cast<size_t>(N) * static_cast<size_t>(K);

    auto rss_kb = []() -> long {
#if !defined(_WIN32)
        struct rusage ru {};
        getrusage(RUSAGE_SELF, &ru);
        return ru.ru_maxrss;
#else
        return 0;
#endif
    };

    std::vector<int8_t> wnk(plain, 1);
    std::vector<float> scl(static_cast<size_t>(ng) * N, 0.125f);

    auto run_one = [&](const char *label, data_type_t wei_dt, int ggml_blk,
                           auto pack_fn, size_t packed_bytes) {
        std::vector<uint8_t> packed(packed_bytes);
        pack_fn(wnk.data(), scl.data(), N, K, packed.data());
        clear_ggml_weight_unpack_cache();
        const long rss0 = rss_kb();
        matmul_params params;
        params.dtypes.wei = wei_dt;
        params.packing.pack_format_b = 1;
        const void *weight = packed.data();
        ASSERT_EQ(unpack_ggml_weights_and_cache(
                          weight, N, K, /*ldb=*/K, 't', params),
                status_t::success)
                << label;
        const long rss1 = rss_kb();
        const size_t cap = params.wei_buffer_capacity_bytes;
        const double mib = 1024.0 * 1024.0;
        std::printf(
                "[GgmlMem] %s N=%d K=%d plain=%.3f MiB cap=%.3f MiB "
                "(pad=+%.3f) wc=%d rss_delta=%.1f MiB "
                "(maxrss %ld -> %ld KiB)\n",
                label, N, K, plain / mib, cap / mib,
                (cap > plain ? (cap - plain) : 0) / mib,
                params.weight_cache_type, (rss1 - rss0) / 1024.0, rss0, rss1);
        EXPECT_EQ(params.packing.pack_format_b, 0) << label;
        EXPECT_EQ(params.weight_cache_type, 2) << label;
        EXPECT_EQ(params.dtypes.wei, wei_dt) << label;
        if (wei_dt == data_type_t::s4) {
            EXPECT_GE(cap, plain / 2) << label;
        } else {
            EXPECT_GE(cap, plain) << label;
            // CK pack for N%32==0: slightly above plain (compensation row).
            const int pack_nr = 32;
            const int K_quad = (K + 3) / 4;
            const size_t ck_bytes = static_cast<size_t>(N / pack_nr)
                    * (static_cast<size_t>(K_quad) * pack_nr * 4
                            + static_cast<size_t>(pack_nr) * sizeof(int32_t));
            EXPECT_GE(cap, ck_bytes) << label;
        }
        clear_ggml_weight_unpack_cache();
        (void)ggml_blk;
    };

    run_one("Q4_0", data_type_t::s4, 2, repack_weights_q4_0,
            static_cast<size_t>(N) * ng * 18);
    run_one("Q8_0", data_type_t::s8, 8, repack_weights_q8_0,
            static_cast<size_t>(N) * ng * 34);
}

/** @brief Q4_0 superblock (block_q4_0x8) stays packed signed s4.
 *  is_superblock is what selects the interleaved layout; decoding that blob
 *  as plain block_q4_0 must not match.
 */
TEST(GgmlWeightUnpackQ4_0, SuperblockKeepsPackedS4) {
    const int N = 8;
    const int K = 64;
    const int ng = K / 32;

    std::vector<int8_t> wnk(static_cast<size_t>(N) * K);
    for (int i = 0; i < N * K; ++i) {
        wnk[i] = static_cast<int8_t>(static_cast<int>((i * 7 + 3) % 16) - 8);
    }
    std::vector<float> scl(static_cast<size_t>(ng) * N);
    for (int i = 0; i < ng * N; ++i)
        scl[i] = 0.125f * static_cast<float>(i + 1);

    std::vector<uint8_t> packed(static_cast<size_t>(N / 8) * ng * 144);
    repack_weights_q4_0_superblock(wnk.data(), scl.data(), N, K, packed.data());

    clear_ggml_weight_unpack_cache();
    matmul_params params;
    params.dtypes.wei = data_type_t::s4;
    params.packing.pack_format_b = 1;
    params.packing.is_superblock = true;

    const void *weight = static_cast<const void *>(packed.data());
    ASSERT_EQ(
            unpack_ggml_weights_and_cache(weight, N, K, /*ldb=*/K, 't', params),
            status_t::success);
    EXPECT_FALSE(params.packing.is_superblock);
    EXPECT_EQ(params.weight_cache_type, 2);
    EXPECT_EQ(params.dtypes.wei, data_type_t::s4);
    EXPECT_GE(params.wei_buffer_capacity_bytes,
            static_cast<size_t>(N) * static_cast<size_t>(K / 2));

    const auto *nibbles = static_cast<const uint8_t *>(weight);
    for (int i = 0; i < N * K; ++i) {
        ASSERT_EQ(packed_s4_code(nibbles, i), static_cast<int>(wnk[i]))
                << "Q4_0 superblock packed-s4 mismatch at index " << i;
    }

    clear_ggml_weight_unpack_cache();
    matmul_params plain;
    plain.dtypes.wei = data_type_t::s4;
    plain.packing.pack_format_b = 1;
    const void *misread = static_cast<const void *>(packed.data());
    ASSERT_EQ(
            unpack_ggml_weights_and_cache(misread, N, K, /*ldb=*/K, 't', plain),
            status_t::success);
    const auto *wrong = static_cast<const uint8_t *>(misread);
    int mismatches = 0;
    for (int i = 0; i < N * K; ++i) {
        if (packed_s4_code(wrong, i) != static_cast<int>(wnk[i])) ++mismatches;
    }
    EXPECT_GT(mismatches, 0)
            << "plain Q4_0 decode of a superblock blob should not round-trip";
    clear_ggml_weight_unpack_cache();
}

/** @brief llama.cpp enters with an f32 activation and dynamic_quant set.
 *  Unpack must run from the packed weight alone; the source is quantized
 *  afterwards.  Q8_0 and Q4_0 superblock both take that entry.
 */
TEST(GgmlPackedLlamaEntry, F32DynamicQuantUnpacksThenMatmul) {
    const int M = 4;
    const int N = 32;
    const int K = 64;
    const int ng = K / 32;

    auto run = [&](const char *label, data_type_t wei_dt, bool superblock,
                       auto pack_fn, size_t packed_bytes) {
        std::vector<int8_t> wnk(static_cast<size_t>(N) * K);
        for (int i = 0; i < N * K; ++i) {
            wnk[i] = static_cast<int8_t>(
                    static_cast<int>((i * 7 + 3) % 16) - 8);
        }
        std::vector<float> scl(static_cast<size_t>(ng) * N, 0.25f);
        std::vector<uint8_t> packed(packed_bytes);
        pack_fn(wnk.data(), scl.data(), N, K, packed.data());

        std::vector<float> src(static_cast<size_t>(M) * K, 0.5f);
        std::vector<float> dst(static_cast<size_t>(M) * N, 0.f);

        matmul_params params;
        params.dtypes.src = data_type_t::f32;
        params.dtypes.wei = wei_dt;
        params.dtypes.dst = data_type_t::f32;
        params.dtypes.compute = data_type_t::s8;
        params.dynamic_quant = true;
        params.quant_params.src_scale.dt = data_type_t::bf16;
        params.quant_params.src_scale.dims
                = {static_cast<int64_t>(M), static_cast<int64_t>(ng)};
        params.packing.pack_format_b = 1;
        params.packing.is_superblock = superblock;
        params.num_threads = 4;

        clear_ggml_weight_unpack_cache();
        clear_matmul_test_caches();
        matmul_batch_params_t batch;
        const status_t st = run_matmul_direct('r', false, true, M, N, K, 1.f,
                src.data(), K, packed.data(), K, nullptr, 0.f, dst.data(), N,
                true, batch, params);
        if (!native_refusal_verified()) {
            EXPECT_EQ(st, status_t::success) << label;
            // Constant 0.5 source, weight codes summing to -32, scale 0.25.
            // Every output is -4 before bf16 scale rounding.
            for (float v : dst) {
                EXPECT_NEAR(v, -4.f, 1e-2f) << label;
            }
        }
        clear_ggml_weight_unpack_cache();
        clear_matmul_test_caches();
    };

    run("Q8_0", data_type_t::s8, false, repack_weights_q8_0,
            static_cast<size_t>(N) * ng * 34);
    run("Q4_0_superblock", data_type_t::s4, true,
            repack_weights_q4_0_superblock,
            static_cast<size_t>(N / 8) * ng * 144);
}

TEST(GgmlPackedValidation, RejectsAsymmetricU8Compute) {
    matmul_params params;
    params.dtypes.src = data_type_t::f32;
    params.dtypes.wei = data_type_t::s4;
    params.dtypes.dst = data_type_t::f32;
    params.dynamic_quant = true;
    params.quant_params.src_scale.dt = data_type_t::bf16;
    params.quant_params.src_scale.dims = {4, 2};

    params.dtypes.compute = data_type_t::s8;
    EXPECT_TRUE(zendnnl::lowoha::matmul::ggml_packed_gemm_supported(params));

    params.dtypes.compute = data_type_t::u8;
    EXPECT_FALSE(zendnnl::lowoha::matmul::ggml_packed_gemm_supported(params))
            << "GGML per-group quantization is symmetric-only";
}

/** @brief Superblock is Q4_0-only and needs 8-row groups. */
TEST(GgmlWeightUnpackQ4_0, SuperblockRejectsBadShapeAndDtype) {
    const int K = 64;
    std::vector<uint8_t> packed(static_cast<size_t>(8) * (K / 32) * 18, 0);

    clear_ggml_weight_unpack_cache();
    matmul_params odd_rows;
    odd_rows.dtypes.wei = data_type_t::s4;
    odd_rows.packing.pack_format_b = 1;
    odd_rows.packing.is_superblock = true;
    const void *weight = static_cast<const void *>(packed.data());
    EXPECT_EQ(unpack_ggml_weights_and_cache(
                      weight, /*N=*/6, K, /*ldb=*/K, 't', odd_rows),
            status_t::failure);

    matmul_params q8;
    q8.dtypes.wei = data_type_t::s8;
    q8.packing.pack_format_b = 1;
    q8.packing.is_superblock = true;
    EXPECT_EQ(unpack_ggml_weights_and_cache(
                      weight, /*N=*/8, K, /*ldb=*/K, 't', q8),
            status_t::failure);
    clear_ggml_weight_unpack_cache();
}

/** @brief GGML packed weights must reject mish at validation.
 *
 *  A Q4_0 weight is still s4 when validate_matmul_direct_inputs runs, so
 *  is_sym_quant_config does not recognise it — but the unpack widens it to s8
 *  and the call lands on the AOCL sym_quant kernel, which silently drops mish.
 *  The gate therefore has to key off pack_format_b, otherwise the caller gets
 *  a wrong result instead of an error (the Q8_0 path already errors out).
 */
TEST(GgmlPackedValidation, MishRejectedOnPackedWeightPath) {
    const int M = 4, N = 8, K = 64;
    std::vector<int8_t> src(static_cast<size_t>(M) * K, 0);
    // Real block_q4_0 layout: N*(K/32) blocks of 18 bytes (fp16 scale + 16
    // packed-nibble bytes), not the raw N*K/2 nibble count.
    std::vector<uint8_t> weight(static_cast<size_t>(N) * (K / 32) * 18, 0);
    std::vector<uint16_t> dst(static_cast<size_t>(M) * N, 0);

    matmul_post_op mish;
    mish.po_type = post_op_type_t::mish;

    matmul_params ggml_params;
    ggml_params.dtypes.src = data_type_t::s8;
    ggml_params.dtypes.wei = data_type_t::s4;
    ggml_params.dtypes.dst = data_type_t::bf16;
    ggml_params.packing.pack_format_b = 1;
    ggml_params.postop_.push_back(mish);
    EXPECT_EQ(validate_matmul_direct_inputs(src.data(), weight.data(),
                      dst.data(), M, N, K, 1, 1, ggml_params,
                      /*is_weights_const=*/true),
            status_t::failure);

    // The same post-op on a plain bf16 call still validates: the gate targets
    // the INT8 kernels, not mish itself.
    matmul_params bf16_params;
    bf16_params.dtypes.src = data_type_t::bf16;
    bf16_params.dtypes.wei = data_type_t::bf16;
    bf16_params.dtypes.dst = data_type_t::bf16;
    bf16_params.postop_.push_back(mish);
    EXPECT_EQ(validate_matmul_direct_inputs(src.data(), weight.data(),
                      dst.data(), M, N, K, 1, 1, bf16_params,
                      /*is_weights_const=*/true),
            status_t::success);
}

/** @brief The GGML unpack must reject weight dtypes it cannot decode.
 *
 *  The block stride comes from params.dtypes.wei, so silently treating
 *  anything non-s4 as Q8_0 would read past the end of the packed blob.
 *  Cold-expert (M==0) cache warming reaches the unpack without the
 *  ggml_is_sym_quant gate, so this check is all that stands between a mis-set
 *  dtype and an out-of-bounds read.
 */
TEST(GgmlPackedValidation, UnsupportedWeightDtypeRejected) {
    const int N = 6;
    const int K = 64;
    std::vector<uint8_t> packed(static_cast<size_t>(N) * (K / 32) * 18, 0);

    clear_ggml_weight_unpack_cache();
    for (data_type_t wei_dt :
            {data_type_t::bf16, data_type_t::f32, data_type_t::u4}) {
        matmul_params params;
        params.dtypes.wei = wei_dt;
        params.packing.pack_format_b = 1;

        const void *weight = static_cast<const void *>(packed.data());
        EXPECT_EQ(unpack_ggml_weights_and_cache(
                          weight, N, K, /*ldb=*/K, 't', params),
                status_t::failure)
                << "wei dtype " << dtype_info(wei_dt) << " should be rejected";
        EXPECT_EQ(weight, static_cast<const void *>(packed.data()))
                << "a rejected unpack must not redirect the weight pointer";
    }
    clear_ggml_weight_unpack_cache();
}

/** @brief Test INT8 sym_quant: per-token source scale, bf16 output */
TEST_P(TestMatmul, INT8_SYM_QUANT_PER_TOKEN_BF16) {
    uint64_t sym_k = (k / 4) * 4;
    if (sym_k == 0) { sym_k = 4; }

    source_dtype = data_type_t::s8;
    use_LOWOHA = true;
    neutralize_mish_quant_int8(po_types);

    data_type_t ref_dt = data_type_t::bf16;
    std::mt19937 local_rng(m ^ k ^ n ^ 0xBF17);
    data_type_t scale_dt
            = (local_rng() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;
    std::vector<int64_t> wei_sd = {1, static_cast<int64_t>(n)};
    std::vector<int64_t> src_sd = {static_cast<int64_t>(m), 1};

    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {sym_k, n}, ref_dt, 25.0, transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor)
            != status_t::success) {
        FAIL() << "weight dynamic quantization failed";
    }

    auto src_ref = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, ref_dt, 25.0, transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                src_sd, scale_dt, src_scale, src_zp, &input_tensor)
            != status_t::success) {
        FAIL() << "source dynamic quantization failed";
    }

    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);

    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0, 0.0, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0, 0.0);
    if (native_refusal_verified()) return;
    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                rtol_bf16, epsilon_bf16, ok, false, 1.0f, true, 0.0f, &po_types,
                &binary_tensors);
    }
    EXPECT_TRUE(ok);
}

/** @brief Test INT8 sym_quant: per-token source scale, f32 output */
TEST_P(TestMatmul, INT8_SYM_QUANT_PER_TOKEN_F32) {
    uint64_t sym_k = (k / 4) * 4;
    if (sym_k == 0) { sym_k = 4; }

    source_dtype = data_type_t::s8;
    use_LOWOHA = true;
    neutralize_mish_quant_int8(po_types);

    data_type_t ref_dt = data_type_t::f32;
    std::mt19937 local_rng(m ^ k ^ n ^ 0xF321);
    data_type_t scale_dt
            = (local_rng() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;
    std::vector<int64_t> wei_sd = {1, static_cast<int64_t>(n)};
    std::vector<int64_t> src_sd = {static_cast<int64_t>(m), 1};

    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {sym_k, n}, ref_dt, 25.0, transB);
    tensor_t weight_tensor, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, wei_ref, ref_dt, data_type_t::s8,
                wei_sd, scale_dt, wei_scale, wei_zp, &weight_tensor)
            != status_t::success) {
        FAIL() << "weight dynamic quantization failed";
    }

    auto src_ref = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, ref_dt, 25.0, transA);
    tensor_t input_tensor, src_scale, src_zp;
    if (quant_params_compute(tensor_factory, src_ref, ref_dt, data_type_t::s8,
                src_sd, scale_dt, src_scale, src_zp, &input_tensor)
            != status_t::success) {
        FAIL() << "source dynamic quantization failed";
    }

    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, data_type_t::f32, 2.0);

    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0, 0.0, true);
    status_t status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0, 0.0);
    if (native_refusal_verified()) return;
    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                rtol_f32, epsilon_f32, ok, false, 1.0f, true, 0.0f, &po_types,
                &binary_tensors);
    }
    EXPECT_TRUE(ok);
}

/** @brief Test INT8 dynamic GEMM: bf16 src, s8 weight (dynamically quantized), bf16 dst vs bf16 reference */
TEST_P(TestMatmul, INT8_DYNAMIC_GEMM_BF16) {
    uint64_t sym_k = (k / 4) * 4;
    if (sym_k == 0) { sym_k = 4; }

    use_LOWOHA = true;
    neutralize_mish_quant_int8(po_types);
    data_type_t test_dt = data_type_t::bf16;

    std::mt19937 local_rng(static_cast<uint32_t>(m ^ (k << 3) ^ (n << 7)
            ^ (static_cast<uint64_t>(transA) << 1)
            ^ (static_cast<uint64_t>(transB) << 2)
            ^ (static_cast<uint64_t>(algo) << 4) ^ (po_types.size() << 8)
            ^ 0xBF18));
    std::vector<uint64_t> valid_gs;
    for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
        if (sym_k % gs == 0) { valid_gs.push_back(gs); }
    }
    // 0: per-token src + per-channel wei
    // 1: per-group src + per-group wei
    // 2: per-token src + per-group wei (newly supported mixed pairing)
    int pairing = static_cast<int>(local_rng() % 3);
    if (valid_gs.empty()) { pairing = 0; }
    uint64_t group_size = 0;
    uint64_t num_groups = 0;
    if (pairing != 0) {
        group_size = valid_gs[local_rng() % valid_gs.size()];
        num_groups = sym_k / group_size;
    }
    const bool wei_per_group = pairing != 0;
    const bool src_per_group = pairing == 1;

    data_type_t scale_dt
            = (local_rng() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;

    std::vector<int64_t> wei_scale_dims = wei_per_group
            ? std::vector<int64_t> {static_cast<int64_t>(num_groups),
                      static_cast<int64_t>(n)}
            : std::vector<int64_t> {1, static_cast<int64_t>(n)};
    std::vector<uint64_t> src_scale_shape = src_per_group
            ? std::vector<uint64_t> {m, num_groups}
            : std::vector<uint64_t> {m, 1};

    auto weight_tensor_ref
            = tensor_factory.uniform_dist_tensor({sym_k, n}, test_dt, 2.0);
    tensor_t weight_tensor_s8, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, weight_tensor_ref, test_dt,
                data_type_t::s8, wei_scale_dims, scale_dt, wei_scale, wei_zp,
                &weight_tensor_s8)
            != status_t::success) {
        FAIL() << "weight dynamic quantization failed";
    }

    auto src_scale = tensor_factory.zero_tensor(src_scale_shape, scale_dt);
    auto input_tensor = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, test_dt, 2.0, transA, src_scale, tensor_t());
    auto input_tensor_ref = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, test_dt, 2.0, transA);
    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);

    const char *pairing_str = pairing == 0
            ? "per-token-src/per-channel-wei"
            : (pairing == 1 ? "per-group-src/per-group-wei"
                            : "per-token-src/per-group-wei");
    log_info("INT8_DYNAMIC_GEMM_BF16: ", pairing_str,
            wei_per_group ? " group_size=" + std::to_string(group_size) : "",
            " scale_dt=", scale_dt == data_type_t::f32 ? "f32" : "bf16");
    SCOPED_TRACE(pairing_str);

    status_t status = matmul_kernel_test(input_tensor, weight_tensor_s8,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0, 0.0);
    if (native_refusal_verified()) return;
    status_t ref_status = matmul_kernel_test(input_tensor_ref,
            weight_tensor_ref, bias_tensor, output_tensor_ref, po_types,
            binary_tensors, use_LOWOHA, algo, 1.0, 0.0, true);
    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
        uint32_t binary_mul_count = 0;
        for (auto po : po_types) {
            if (po == post_op_type_t::binary_mul) { ++binary_mul_count; }
        }
        // 18x base for dynamic quant; scale by max(1, 2*binary_mul_count) for binary_mul post-ops.
        const uint32_t binary_mul_scale
                = binary_mul_count ? (2u * binary_mul_count) : 1u;
        const float epsilon
                = static_cast<float>(binary_mul_scale) * 18.f * epsilon_bf16;
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                rtol_bf16, epsilon, ok, false, 1.0f, true);
    }
    EXPECT_TRUE(ok);
}

/** @brief Test INT8 dynamic GEMM: f32 src, s8 weight (dynamically quantized), f32 dst vs f32 reference */
TEST_P(TestMatmul, INT8_DYNAMIC_GEMM_F32) {
    uint64_t sym_k = (k / 4) * 4;
    if (sym_k == 0) { sym_k = 4; }

    use_LOWOHA = true;
    neutralize_mish_quant_int8(po_types);
    data_type_t test_dt = data_type_t::f32;

    std::mt19937 local_rng(static_cast<uint32_t>(m ^ (k << 3) ^ (n << 7)
            ^ (static_cast<uint64_t>(transA) << 1)
            ^ (static_cast<uint64_t>(transB) << 2)
            ^ (static_cast<uint64_t>(algo) << 4) ^ (po_types.size() << 8)
            ^ 0xF322));
    std::vector<uint64_t> valid_gs;
    for (uint64_t gs = 4; gs <= sym_k; gs *= 2) {
        if (sym_k % gs == 0) { valid_gs.push_back(gs); }
    }
    // 0: per-token src + per-channel wei
    // 1: per-group src + per-group wei
    // 2: per-token src + per-group wei (newly supported mixed pairing)
    int pairing = static_cast<int>(local_rng() % 3);
    if (valid_gs.empty()) { pairing = 0; }
    uint64_t group_size = 0;
    uint64_t num_groups = 0;
    if (pairing != 0) {
        group_size = valid_gs[local_rng() % valid_gs.size()];
        num_groups = sym_k / group_size;
    }
    const bool wei_per_group = pairing != 0;
    const bool src_per_group = pairing == 1;

    data_type_t scale_dt
            = (local_rng() % 2 == 0) ? data_type_t::f32 : data_type_t::bf16;

    std::vector<int64_t> wei_scale_dims = wei_per_group
            ? std::vector<int64_t> {static_cast<int64_t>(num_groups),
                      static_cast<int64_t>(n)}
            : std::vector<int64_t> {1, static_cast<int64_t>(n)};
    std::vector<uint64_t> src_scale_shape = src_per_group
            ? std::vector<uint64_t> {m, num_groups}
            : std::vector<uint64_t> {m, 1};

    auto weight_tensor_ref
            = tensor_factory.uniform_dist_tensor({sym_k, n}, test_dt, 2.0);
    tensor_t weight_tensor_s8, wei_scale, wei_zp;
    if (quant_params_compute(tensor_factory, weight_tensor_ref, test_dt,
                data_type_t::s8, wei_scale_dims, scale_dt, wei_scale, wei_zp,
                &weight_tensor_s8)
            != status_t::success) {
        FAIL() << "weight dynamic quantization failed";
    }

    auto src_scale = tensor_factory.zero_tensor(src_scale_shape, scale_dt);
    auto input_tensor = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, test_dt, 2.0, transA, src_scale, tensor_t());
    auto input_tensor_ref = tensor_factory.uniform_dist_tensor(
            {m, sym_k}, test_dt, 2.0, transA);
    auto bias_tensor = tensor_factory.uniform_dist_tensor({1, n},
            rand() % 2 == 0 ? data_type_t::bf16 : data_type_t::f32, 2.0);
    auto binary_tensor_shape_2d = {m, n};
    auto binary_tensor_shape_broadcast = {uint64_t {1}, n};
    auto binary_tensor_shape = (rand() % 2 == 0) ? binary_tensor_shape_broadcast
                                                 : binary_tensor_shape_2d;
    auto binary_tensors = make_binary_postop_tensors(
            tensor_factory, po_types, binary_tensor_shape);
    auto output_tensor
            = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);
    auto output_tensor_ref
            = tensor_factory.uniform_dist_tensor({m, n}, test_dt, 2.0);

    const char *pairing_str = pairing == 0
            ? "per-token-src/per-channel-wei"
            : (pairing == 1 ? "per-group-src/per-group-wei"
                            : "per-token-src/per-group-wei");
    log_info("INT8_DYNAMIC_GEMM_F32: ", pairing_str,
            wei_per_group ? " group_size=" + std::to_string(group_size) : "",
            " scale_dt=", scale_dt == data_type_t::f32 ? "f32" : "bf16");
    SCOPED_TRACE(pairing_str);

    status_t status = matmul_kernel_test(input_tensor, weight_tensor_s8,
            bias_tensor, output_tensor, po_types, binary_tensors, use_LOWOHA,
            algo, 1.0, 0.0);
    if (native_refusal_verified()) return;
    status_t ref_status = matmul_kernel_test(input_tensor_ref,
            weight_tensor_ref, bias_tensor, output_tensor_ref, po_types,
            binary_tensors, use_LOWOHA, algo, 1.0, 0.0, true);
    bool ok = (status == status_t::success && ref_status == status_t::success);
    if (ok) {
        uint32_t binary_mul_count = 0;
        for (auto po : po_types) {
            if (po == post_op_type_t::binary_mul) { ++binary_mul_count; }
        }
        // 18x base for dynamic quant; scale by max(1, 2*binary_mul_count) for binary_mul post-ops.
        const uint32_t binary_mul_scale
                = binary_mul_count ? (2u * binary_mul_count) : 1u;
        const float epsilon
                = static_cast<float>(binary_mul_scale) * 18.f * epsilon_bf16;
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, sym_k,
                rtol_bf16, epsilon, ok, false, 1.0f, true);
    }
    EXPECT_TRUE(ok);
}

/** @fn TEST_P
 *  @param TestMatmul parameterized test class to initialize Matmul parameters
 *  @param BF16_INT8 user-defined name of test according to test
 *  @brief Test to validate matmul INT8 aocl kernel support wrt Reference kernel
 */
//  TEST_P(TestMatmul, BF16_INT8) {

//   // if (true) {
//   //   GTEST_SKIP();
//   // }
//   output_dtype = data_type_t::bf16;
//   source_dtype = data_type_t::bf16;
//   use_LOWOHA = true;
//   algo = algo != matmul_algo_t::aocl_dlp ? matmul_algo_t::aocl_dlp_blocked : algo;

//   bool is_u8_source       = false;// Currently, only s8 source conversion is supported by DLP
//   auto wei_scale          = tensor_factory.uniform_dist_tensor({1, 1},
//                             data_type_t::f32, 0.6);
//   auto src_scale          = tensor_factory.uniform_tensor({1, 1},
//                             data_type_t::f32, 0.2);
//   auto src_zp             = is_u8_source ? tensor_factory.uniform_tensor({1, 1},
//                             data_type_t::s32, 16) : tensor_t();

//   auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
//                             data_type_t::s8, 25.0, transB, wei_scale);
//   auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
//                             source_dtype, 2.0, transA, src_scale, src_zp);
//   auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, data_type_t::bf16, 1.0);
//   auto binary_tensor      = is_binary_postop(po_type) ?
//                             tensor_factory.uniform_dist_tensor({m, n},
//                                 data_type_t::f32, 2.0) : tensor_t();
//   auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
//                             output_dtype, 2.0, false);
//   auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
//                             output_dtype, 2.0, false);
//   status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
//                             bias_tensor, output_tensor, po_type, binary_tensor,
//                             use_LOWOHA, algo, 1.0, 0.0);
//   status_t ref_status     = matmul_kernel_test(input_tensor,
//                             weight_tensor, bias_tensor, output_tensor_ref, po_type,
//                             binary_tensor, use_LOWOHA, algo, 1.0, 0.0, true);

//   bool is_test_successful =
//     (status == status_t::success && ref_status == status_t::success);
//   if (is_test_successful) {
//     compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m,n,k, rtol_bf16,
//                              epsilon_bf16, is_test_successful);
//   }

//   EXPECT_TRUE(is_test_successful);
// }

// /** @fn TEST_P
//  *  @param TestMatmul parameterized test class to initialize Matmul parameters
//  *  @param F32_INT8 user-defined name of test according to test
//  *  @brief Test to validate matmul INT8 aocl kernel support wrt Reference kernel
//  */
//  TEST_P(TestMatmul, F32_INT8) {

//   // if (true) {
//   //   GTEST_SKIP();
//   // }
//   output_dtype = data_type_t::f32;
//   source_dtype = data_type_t::f32;
//   use_LOWOHA = true;
//   algo = algo != matmul_algo_t::aocl_dlp ? matmul_algo_t::aocl_dlp_blocked : algo;

//   bool is_u8_source       = false;// Currently, only s8 source conversion is supported by DLP
//   auto wei_scale          = tensor_factory.uniform_dist_tensor({1, n},
//                             data_type_t::f32, 0.6);
//   auto src_scale          = tensor_factory.uniform_tensor({1, 1},
//                             data_type_t::f32, 0.2);
//   auto src_zp             = is_u8_source ? tensor_factory.uniform_tensor({1, 1},
//                             data_type_t::s32, 16) : tensor_t();

//   auto weight_tensor      = tensor_factory.uniform_dist_tensor({k, n},
//                             data_type_t::s8, 25.0, transB, wei_scale);
//   auto input_tensor       = tensor_factory.uniform_dist_tensor({m, k},
//                             source_dtype, 2.0, transA, src_scale, src_zp);
//   auto bias_tensor        = tensor_factory.uniform_dist_tensor({1, n}, data_type_t::bf16, 1.0);
//   auto binary_tensor      = is_binary_postop(po_type) ?
//                             tensor_factory.uniform_dist_tensor({m, n},
//                                 data_type_t::f32, 2.0) : tensor_t();
//   auto output_tensor      = tensor_factory.uniform_dist_tensor({m, n},
//                             output_dtype, 2.0, false);
//   auto output_tensor_ref  = tensor_factory.uniform_dist_tensor({m, n},
//                             output_dtype, 2.0, false);
//   status_t status         = matmul_kernel_test(input_tensor, weight_tensor,
//                             bias_tensor, output_tensor, po_type, binary_tensor,
//                             use_LOWOHA, algo, 1.0, 0.0);
//   status_t ref_status     = matmul_kernel_test(input_tensor,
//                             weight_tensor, bias_tensor, output_tensor_ref, po_type,
//                             binary_tensor, use_LOWOHA, algo, 1.0, 0.0, true);

//   bool is_test_successful =
//     (status == status_t::success && ref_status == status_t::success);
//   if (is_test_successful) {
//     compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m,n,k, rtol_bf16,
//                              epsilon_bf16, is_test_successful);
//   }

//   EXPECT_TRUE(is_test_successful);
// }

// Post-op metadata cache tests for the AOCL DLP backend live in
// test_postop_cache.cpp (their own value-parameterized fixture).

/** @fn INSTANTIATE_TEST_SUITE_P
 *  @brief Triggers Matmul parameterized test suite
 */
INSTANTIATE_TEST_SUITE_P(Matmul, TestMatmul, ::testing::ValuesIn(matmul_test));
