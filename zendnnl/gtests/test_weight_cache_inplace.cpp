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

// =============================================================================
// In-place weight cache vs. INT8 zero-point compensation.
//
// Under ZENDNNL_MATMUL_WEIGHT_CACHE=2 the AOCL-DLP blocked matmul reorders the
// caller's weight buffer IN PLACE on the first call and reuses the mutated
// buffer on later calls. The INT8 zero-point compensation is derived from the
// PLAIN (pre-reorder) weights, so if it is (re)computed after the in-place
// reorder it reads scrambled bytes and miscomputes.
//
// This test drives the full u8*s8 quantized matmul through the same harness as
// test_matmul.cpp (matmul_kernel_test, which does a warmup + real call on the
// SAME weight buffer, then matmul_kernel_test(..., use_reference=true) for the
// golden baseline),
// pinned to WEIGHT_CACHE=2 so the in-place reorder path is exercised. It covers
// the common case: u8 activations (nonzero src_zp) + symmetric s8 weights
// (wei_zp == 0). The matmul is run several times on the same in-place-reordered
// weight buffer to confirm accuracy holds across cache reuse, not just the
// first warmup+real pair.
//
// The two related vulnerable paths (asymmetric wei_zp != 0, and ZP-comp caching
// disabled) are documented as a TODO next to the compensation code in
// aocl_kernel.cpp (run_dlp).
//
// WeightCacheGuard / ZpCompCacheGuard are shared RAII helpers from
// gtest_utils.hpp.
// =============================================================================

#include <gtest/gtest.h>

#include "common/op_config.hpp"
#include "gtest_utils.hpp"

namespace {

class TestWeightCacheInplace : public ::testing::Test {
protected:
    void TearDown() override { clear_matmul_test_caches(); }
    tensor_factory_t tensor_factory {};
};

} // namespace

// Common case: u8 activations (nonzero src_zp) x symmetric s8 weights
// (wei_zp == 0) -> 1D zero-point compensation. Run under WEIGHT_CACHE=2 (in-place
// reorder + weight-buffer reuse across the warmup and real call) and compare
// against the reference kernel. The 1D compensation is computed from the plain
// weights and cached, so the in-place reorder cannot corrupt it.
TEST_F(TestWeightCacheInplace, Cached1D_InPlaceWeightMutation) {
    const uint64_t m = 32, k = 256, n = 256;
    const bool use_LOWOHA = true;
    const auto algo = matmul_algo_t::aocl_dlp_blocked;
    const std::vector<post_op_type_t> po_types {};
    const std::vector<tensor_t> binary_tensors {};

    // Symmetric s8 weights (per-tensor scale, wei_zp == 0).
    auto wei_ref = tensor_factory.uniform_dist_tensor(
            {k, n}, data_type_t::bf16, 25.0, /*transB=*/false);
    tensor_t weight_tensor, wei_scale, wei_zp;
    ASSERT_EQ(quant_params_compute(tensor_factory, wei_ref, data_type_t::bf16,
                      data_type_t::s8, {1, 1}, data_type_t::f32, wei_scale,
                      wei_zp, &weight_tensor),
            status_t::success)
            << "weight quantization failed";

    // u8 activations -> asymmetric -> nonzero src_zp.
    auto src_ref = tensor_factory.uniform_dist_tensor(
            {m, k}, data_type_t::bf16, 25.0, /*transA=*/false);
    tensor_t input_tensor, src_scale, src_zp;
    ASSERT_EQ(quant_params_compute(tensor_factory, src_ref, data_type_t::bf16,
                      data_type_t::u8, {1, 1}, data_type_t::f32, src_scale,
                      src_zp, &input_tensor),
            status_t::success)
            << "source quantization failed";

    auto bias_tensor
            = tensor_factory.uniform_dist_tensor({1, n}, data_type_t::f32, 2.0);
    auto output_tensor = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);
    auto output_tensor_ref = tensor_factory.uniform_dist_tensor(
            {m, n}, data_type_t::bf16, 2.0);

    // Reference FIRST, on the pristine weight buffer: the WEIGHT_CACHE=2 run below
    // reorders those weights IN PLACE, so computing the reference afterwards would
    // read the mutated (blocked) bytes and produce a wrong baseline.
    status_t ref_status = matmul_kernel_test(input_tensor, weight_tensor,
            bias_tensor, output_tensor_ref, po_types, binary_tensors,
            use_LOWOHA, algo, 1.0, 0.0, true);
    ASSERT_EQ(ref_status, status_t::success) << "reference kernel failed";
    clear_matmul_test_caches();

    // Pin WEIGHT_CACHE=2 (in-place reorder + weight-buffer reuse) and force the
    // 1D ZP-comp cache ON so the assertion below is not perturbed by an external
    // ZENDNNL_ZP_COMP_CACHE=0 in the environment. Both are restored on scope exit.
    WeightCacheGuard wc_guard(2);
    ZpCompCacheGuard zp_guard(true);

    // Run the matmul repeatedly on the SAME weight buffer. matmul_kernel_test
    // warms up (reordering the weights in place on the first touch) and runs the
    // real matmul; subsequent iterations reuse the in-place-reordered weights and
    // the cached 1D compensation. Every iteration must match the reference — if
    // the compensation were derived from the reordered bytes, reuse would diverge.
    constexpr int kIterations = 3;
    for (int iter = 0; iter < kIterations; ++iter) {
        status_t status = matmul_kernel_test(input_tensor, weight_tensor,
                bias_tensor, output_tensor, po_types, binary_tensors,
                use_LOWOHA, algo, 1.0, 0.0);
        if (status == status_t::isa_unsupported) {
            GTEST_SKIP() << "AOCL-DLP blocked INT8 not supported on this ISA";
        }
        ASSERT_EQ(status, status_t::success)
                << "matmul failed on iteration " << iter;

        bool ok = true;
        compare_tensor_2D_matrix(output_tensor, output_tensor_ref, m, n, k,
                rtol_bf16, epsilon_bf16, ok, false, 1.0f,
                /*is_quant=*/true);
        EXPECT_TRUE(ok)
                << "WEIGHT_CACHE=2 output diverged from the reference on "
                   "iteration "
                << iter
                << " — in-place reorder corrupted "
                   "the zero-point compensation";
    }
}
